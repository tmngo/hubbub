use crate::translate::input::sizeof;
use core::mem;
use std::collections::HashMap;
use std::fmt::Write;
use std::fs;
use std::path::Path;

use crate::analyze::Lookup;
use crate::builtin;
use crate::parse::{Node, NodeId, Tag};
use crate::translate::input::{Data, Input, Layout};
use crate::typecheck::{TypeId, TypeIndex};

use cranelift::codegen;
use cranelift::codegen::ir::StackSlot;
use cranelift::prelude::{isa, settings};
use cranelift::prelude::{
    AbiParam, FunctionBuilder, FunctionBuilderContext, InstBuilder, IntCC, MemFlags, StackSlotData,
    StackSlotKind, Type, Value, Variable,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

pub trait CraneliftModule: Module {
    fn finalize(self: Box<Self>, id: FuncId, output_file: &Path) -> i64;
}

impl CraneliftModule for JITModule {
    fn finalize(mut self: Box<Self>, id: FuncId, _output_file: &Path) -> i64 {
        self.finalize_definitions();
        let main = self.get_finalized_function(id);
        let main: fn() -> i64 = unsafe { mem::transmute(main) };
        println!("--- JIT output:");
        let result = main();
        println!("--- JIT result: {}", result);
        result
    }
}

impl CraneliftModule for ObjectModule {
    fn finalize(self: Box<Self>, _id: FuncId, output_file: &Path) -> i64 {
        let product = self.finish();
        let bytes = product.emit().unwrap();
        fs::write(output_file, bytes).unwrap();
        0
    }
}

pub struct FnContext<'a> {
    b: FunctionBuilder<'a>,
    ptr_type: Type,
    target_config: isa::TargetFrontendConfig,
}

pub struct Generator<'a> {
    pub builder_ctx: FunctionBuilderContext,
    pub ctx: codegen::Context,
    data_ctx: DataContext,
    data: Data<'a>,
    pub state: State,
}

impl<'a> Generator<'a> {
    pub fn new(input: Input<'a>, output_name: String, use_jit: bool) -> Self {
        let flag_builder = settings::builder();
        // flag_builder.enable("is_pic").unwrap();
        let isa_builder = isa::lookup(target_lexicon::HOST).unwrap();
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();

        let module: Box<dyn CraneliftModule> = if use_jit {
            let mut jit_builder =
                JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
            jit_builder.symbols(vec![
                ("Base.print_int", builtin::print_int as *const u8),
                ("Base.alloc", builtin::alloc as *const u8),
                ("Base.dealloc", builtin::dealloc as *const u8),
            ]);
            Box::new(JITModule::new(jit_builder))
        } else {
            let object_builder =
                ObjectBuilder::new(isa, output_name, cranelift_module::default_libcall_names())
                    .unwrap();
            Box::new(ObjectModule::new(object_builder))
        };

        let ty = module.target_config().pointer_type();
        let layouts = input
            .types
            .iter()
            .map(|typ| Layout::new(input.types, &typ, ty))
            .collect();

        Self {
            builder_ctx: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_ctx: DataContext::new(),
            data: Data::new(input, layouts),
            state: State {
                module,
                locations: HashMap::new(),
            },
        }
    }

    ///
    pub fn init_signature(
        func: &mut codegen::ir::Function,
        parameters: &Node,
        returns: &Node,
        t: Type,
    ) {
        for _ in parameters.lhs..parameters.rhs {
            // Tag::Field
            func.signature.params.push(AbiParam::new(t));
        }
        match returns.tag {
            Tag::Expressions => {
                for _ in returns.lhs..returns.rhs {
                    func.signature.returns.push(AbiParam::new(t));
                }
            }
            Tag::Identifier => {
                func.signature.returns.push(AbiParam::new(t));
            }
            _ => {}
        }
    }

    ///
    pub fn compile_nodes(mut self, filename: &Path) -> Option<i64> {
        let root = &self.data.node(0);
        let mut fn_ids = Vec::new();
        let mut main_id = None;

        for i in root.lhs..root.rhs {
            let module_index = self.data.node_index(i);
            let module = self.data.node(module_index);

            if module.tag != Tag::Module {
                continue;
            }

            for i in module.lhs..module.rhs {
                let ni = self.data.node_index(i);
                let node = self.data.node(ni);
                match node.tag {
                    Tag::FunctionDecl => {
                        // Skip generic functions with no specializations.
                        if self.data.node(node.lhs).tag == Tag::ParametricPrototype
                            && !self.data.type_parameters.contains_key(&ni)
                        {
                            continue;
                        }
                        // Skip function signatures
                        if node.rhs == 0 {
                            continue;
                        }
                        let fn_id = Self::compile_function_decl(
                            &self.data,
                            &mut self.ctx,
                            &mut self.builder_ctx,
                            &mut self.state,
                            ni,
                        );
                        let fn_name = self.data.tree.name(ni);
                        if fn_name == "main" {
                            fn_ids.push(fn_id);
                            main_id = Some(fn_id);
                        }
                    }
                    _ => {}
                };
            }
        }
        if let Some(id) = main_id {
            return Some(self.state.module.finalize(id, filename));
        }
        None
    }

    ///
    pub fn compile_function_decl(
        data: &Data,
        ctx: &mut codegen::Context,
        builder_ctx: &mut FunctionBuilderContext,
        state: &mut State,
        node_id: NodeId,
    ) -> FuncId {
        let target_config = state.module.target_config();
        let ptr_type = target_config.pointer_type();
        let mut c = FnContext {
            b: FunctionBuilder::new(&mut ctx.func, builder_ctx),
            ptr_type,
            target_config,
        };

        let node = data.node(node_id);
        Self::compile_function_signature(state, &data, &mut c, node.lhs);
        Self::compile_function_body(state, &data, &mut c, node.rhs);

        c.b.finalize();
        let name = mangle_function_declaration(data, node_id, false);
        println!("{} :: {}", name, c.b.func.display());
        let fn_id = state
            .module
            .declare_function(&name, Linkage::Export, &mut c.b.func.signature)
            .unwrap();
        state.module.define_function(fn_id, ctx).unwrap();
        state.module.clear_context(ctx);
        fn_id
    }

    fn compile_function_signature(
        state: &mut State,
        data: &Data,
        c: &mut FnContext,
        node_id: NodeId,
    ) {
        let prototype = data.node(node_id);
        let parameters = data.node(prototype.lhs);
        let returns = data.node(prototype.rhs);
        Self::init_signature(&mut c.b.func, parameters, returns, c.ptr_type);

        let entry_block = c.b.create_block();
        c.b.append_block_params_for_function_params(entry_block);
        c.b.switch_to_block(entry_block);
        c.b.seal_block(entry_block);

        // Define parameters as stack variables.
        for i in parameters.lhs..parameters.rhs {
            let ni = data.node_index(i);
            let stack_slot = state.create_stack_slot(&data, c, ni);
            let location = Location::stack(stack_slot, 0);
            state.locations.insert(ni, location);
            let layout = data.layout(ni);
            let value = c.b.block_params(entry_block)[(i - parameters.lhs) as usize];
            location.store(c, value, MemFlags::new(), layout);
        }
    }

    fn compile_function_body(state: &mut State, data: &Data, c: &mut FnContext, node_id: NodeId) {
        let body = data.node(node_id);
        for i in body.lhs..body.rhs {
            let ni = data.node_index(i);
            state.compile_stmt(data, c, ni);
        }
    }
}

pub struct State {
    locations: HashMap<NodeId, Location>,
    pub module: Box<dyn CraneliftModule>,
}

impl State {
    ///
    pub fn compile_stmt(&mut self, data: &Data, c: &mut FnContext, node_id: NodeId) {
        let node = data.node(node_id);
        match node.tag {
            Tag::Assign => {
                // lhs: expr
                // rhs: expr
                assert_eq!(data.type_id(node.lhs), data.type_id(node.rhs));
                let layout = data.layout(node.rhs);
                let left_node = data.node(node.lhs);
                let right_value = self
                    .compile_expr(data, c, node.rhs)
                    .cranelift_value(c, layout);
                let flags = MemFlags::new();
                let left_location = match left_node.tag {
                    // a.x = b
                    Tag::Access => self.locate_field(data, node.lhs),
                    // a = b
                    Tag::Identifier => self.locate_variable(data, node.lhs),
                    // @a = b
                    Tag::Dereference => {
                        let ptr_layout = data.layout(left_node.lhs);
                        let ptr = self
                            .locate(data, left_node.lhs)
                            .load_value(c, flags, ptr_layout);
                        Location::pointer(ptr, 0)
                    }
                    _ => unreachable!("Invalid lvalue for assignment"),
                };
                left_location.store(c, right_value, flags, layout);
            }
            Tag::If => {
                let condition_expr = self
                    .compile_expr(data, c, node.lhs)
                    .cranelift_value(c, data.layout(node.lhs));
                let then_block = c.b.create_block();
                let merge_block = c.b.create_block();
                let body = data.node(node.rhs);
                c.b.ins().brz(condition_expr, merge_block, &[]);
                c.b.ins().jump(then_block, &[]);
                c.b.seal_block(then_block);
                // then block
                c.b.switch_to_block(then_block);
                for i in body.lhs..body.rhs {
                    let index = data.node_index(i);
                    self.compile_stmt(data, c, index);
                }
                if !c.b.is_filled() {
                    c.b.ins().jump(merge_block, &[]);
                }
                c.b.seal_block(merge_block);
                // merge block
                c.b.switch_to_block(merge_block);
            }
            Tag::IfElse => {
                let mut if_nodes = Vec::new();
                let mut then_blocks = Vec::new();
                for i in node.lhs..node.rhs {
                    let index = data.node_index(i);
                    let if_node = data.node(index);
                    if_nodes.push(if_node);
                    then_blocks.push(c.b.create_block());
                }
                let if_count = if if_nodes.last().unwrap().lhs == 0 {
                    if_nodes.len() - 1
                } else {
                    if_nodes.len()
                };
                let merge_block = c.b.create_block();
                for i in 0..if_count {
                    let condition_expr = self
                        .compile_expr(data, c, if_nodes[i].lhs)
                        .cranelift_value(c, data.layout(if_nodes[i].lhs));
                    c.b.ins().brnz(condition_expr, then_blocks[i], &[]);
                    c.b.seal_block(then_blocks[i]);
                    if i < if_count - 1 {
                        let block = c.b.create_block();
                        c.b.ins().jump(block, &[]);
                        c.b.seal_block(block);
                        c.b.switch_to_block(block);
                    }
                    if i == if_nodes.len() - 1 {
                        c.b.ins().jump(merge_block, &[]);
                    }
                }
                for i in if_count..if_nodes.len() {
                    c.b.ins().jump(then_blocks[i], &[]);
                    c.b.seal_block(then_blocks[i]);
                }
                for (i, if_node) in if_nodes.iter().enumerate() {
                    c.b.switch_to_block(then_blocks[i]);
                    let body = data.node(if_node.rhs);
                    for j in body.lhs..body.rhs {
                        let index = data.node_index(j);
                        self.compile_stmt(data, c, index);
                    }
                    if !c.b.is_filled() {
                        c.b.ins().jump(merge_block, &[]);
                    }
                }
                c.b.seal_block(merge_block);
                c.b.switch_to_block(merge_block);
            }
            Tag::VariableDecl => {
                // lhs: type
                // rhs: expr
                let slot = self.create_stack_slot(data, c, node_id);
                let location = Location::stack(slot, 0);
                self.locations.insert(node_id, location);
                if node.rhs != 0 {
                    let layout = data.layout(node.rhs);
                    let value = self
                        .compile_expr(data, c, node.rhs)
                        .cranelift_value(c, data.layout(node.rhs));
                    location.store(c, value, MemFlags::new(), layout);
                }
            }
            Tag::Return => {
                let mut return_values = Vec::new();
                for i in node.lhs..node.rhs {
                    let ni = data.node_index(i);
                    let val = self
                        .compile_expr(data, c, ni)
                        .cranelift_value(c, data.layout(ni));
                    return_values.push(val);
                }
                c.b.ins().return_(&return_values[..]);
            }
            Tag::While => {
                let condition = self
                    .compile_expr(data, c, node.lhs)
                    .cranelift_value(c, data.layout(node.lhs));
                let while_block = c.b.create_block();
                let merge_block = c.b.create_block();
                // check condition
                // true? jump to loop body
                c.b.ins().brnz(condition, while_block, &[]);
                // false? jump to after loop
                c.b.ins().jump(merge_block, &[]);
                // block_while:
                c.b.switch_to_block(while_block);
                let body = data.node(node.rhs);
                for i in body.lhs..body.rhs {
                    let ni = data.node_index(i);
                    self.compile_stmt(data, c, ni);
                }
                let condition = self
                    .compile_expr(data, c, node.lhs)
                    .cranelift_value(c, data.layout(node.lhs));
                // brnz block_while
                c.b.ins().brnz(condition, while_block, &[]);
                c.b.seal_block(while_block);
                c.b.ins().jump(merge_block, &[]);
                c.b.seal_block(merge_block);
                // block_merge:
                c.b.switch_to_block(merge_block);
            }
            _ => {
                self.compile_expr(data, c, node_id);
            }
        }
    }

    /// Returns a value. A value can be a scalar or an aggregate.
    /// NodeId -> Val
    pub fn compile_expr(&mut self, data: &Data, c: &mut FnContext, node_id: NodeId) -> Val {
        let ty = c.ptr_type;
        let node = data.node(node_id);
        let layout = data.layout(node_id);
        match node.tag {
            Tag::Access => self.locate_field(data, node_id).to_val(c, layout),
            Tag::Address => Val::Scalar(self.locate(data, node.lhs).get_addr(c)),
            Tag::Dereference => {
                let flags = MemFlags::new();
                let ptr_layout = data.layout(node.lhs);
                let ptr = self.locate(data, node.lhs).load_value(c, flags, ptr_layout);
                Location::pointer(ptr, 0).to_val(c, layout)
            }
            Tag::Add => {
                let (lhs, rhs) = self.compile_children(data, c, node);
                Val::Scalar(c.b.ins().iadd(lhs, rhs))
            }
            Tag::BitwiseShiftL => {
                let (lhs, rhs) = self.compile_children(data, c, node);
                Val::Scalar(c.b.ins().ishl(lhs, rhs))
            }
            Tag::BitwiseShiftR => {
                let (lhs, rhs) = self.compile_children(data, c, node);
                Val::Scalar(c.b.ins().sshr(lhs, rhs))
            }
            Tag::BitwiseXor => {
                let (lhs, rhs) = self.compile_children(data, c, node);
                Val::Scalar(c.b.ins().bxor(lhs, rhs))
            }
            Tag::Sub => {
                let (lhs, rhs) = self.compile_children(data, c, node);
                Val::Scalar(c.b.ins().isub(lhs, rhs))
            }
            Tag::Div => {
                let (lhs, rhs) = self.compile_children(data, c, node);
                Val::Scalar(c.b.ins().sdiv(lhs, rhs))
            }
            Tag::Mul => {
                let (lhs, rhs) = self.compile_children(data, c, node);
                Val::Scalar(c.b.ins().imul(lhs, rhs))
            }
            Tag::Equality => {
                let (lhs, rhs) = self.compile_children(data, c, node);
                Val::Scalar(c.b.ins().icmp(IntCC::Equal, lhs, rhs))
            }
            Tag::Greater => {
                let (lhs, rhs) = self.compile_children(data, c, node);
                Val::Scalar(c.b.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs))
            }
            Tag::Less => {
                let (lhs, rhs) = self.compile_children(data, c, node);
                Val::Scalar(c.b.ins().icmp(IntCC::SignedLessThan, lhs, rhs))
            }
            Tag::Grouping => self.compile_expr(data, c, node.lhs),
            Tag::IntegerLiteral => {
                let token_str = data.node_lexeme_offset(node, 0);
                let value = token_str.parse::<i64>().unwrap();
                Val::Scalar(c.b.ins().iconst(ty, value))
            }
            Tag::True => Val::Scalar(c.b.ins().iconst(ty, 1)),
            Tag::False => Val::Scalar(c.b.ins().iconst(ty, 0)),
            Tag::Call => {
                let mut sig = self.module.make_signature();
                let function = data.node(node.lhs);
                let name = data.tree.name(node.lhs);

                let function_id = data
                    .definitions
                    .get_definition_id(node.lhs, "failed to get function decl");

                println!("name:         {}", name);
                println!(
                    "mangled call: {}",
                    mangle_function_declaration(data, function_id, false)
                );

                let name = mangle_function_declaration(data, function_id, false);

                // Arguments
                let arguments = data.node(node.rhs);
                let mut args = Vec::new();
                for i in arguments.lhs..arguments.rhs {
                    let ni = data.node_index(i);
                    let layout = data.layout(ni);
                    let value = self.compile_expr(data, c, ni).cranelift_value(c, layout);
                    sig.params.push(AbiParam::new(ty));
                    args.push(value);
                }

                // Assume one return value.
                let return_type = data.type_id(node_id);
                if return_type != TypeIndex::Void as TypeId {
                    sig.returns.push(AbiParam::new(ty));
                }

                let callee = self
                    .module
                    .declare_function(&name, Linkage::Import, &sig)
                    .unwrap();
                let local_callee = self.module.declare_func_in_func(callee, c.b.func);
                let call = c.b.ins().call(local_callee, &args);
                if return_type != TypeIndex::Void as TypeId {
                    Val::Scalar(c.b.inst_results(call)[0])
                } else {
                    Val::Scalar(c.b.ins().iconst(ty, 0))
                }
            }
            Tag::Identifier => self.locate(data, node_id).to_val(c, layout),
            _ => unreachable!("Invalid expression tag: {:?}", node.tag),
        }
    }

    fn compile_children(&mut self, data: &Data, c: &mut FnContext, node: &Node) -> (Value, Value) {
        let lhs = self
            .compile_expr(data, c, node.lhs)
            .cranelift_value(c, data.layout(node.lhs));
        let rhs = self
            .compile_expr(data, c, node.rhs)
            .cranelift_value(c, data.layout(node.rhs));
        (lhs, rhs)
    }

    fn locate(&self, data: &Data, node_id: NodeId) -> Location {
        let node = data.node(node_id);
        match node.tag {
            Tag::Access => self.locate_field(data, node_id),
            Tag::Identifier => self.locate_variable(data, node_id),
            _ => unreachable!("Cannot locate node with tag {:?}", node.tag),
        }
    }

    fn locate_variable(&self, data: &Data, node_id: NodeId) -> Location {
        let def_id = data
            .definitions
            .get_definition_id(node_id, "failed to look up variable definition");
        *self.locations.get(&def_id).expect("failed to get location")
    }

    fn locate_field(&self, data: &Data, node_id: NodeId) -> Location {
        let mut indices = Vec::new();
        let mut type_ids = Vec::new();
        let mut parent_id = node_id;
        let mut parent = data.node(parent_id);

        while parent.tag == Tag::Access {
            let field_index = data
                .definitions
                .get_definition_id(parent.rhs, "failed to lookup field definition")
                as usize;
            indices.push(field_index);
            parent_id = parent.lhs;
            type_ids.push(data.type_id(parent_id));
            parent = data.node(parent_id);
        }

        let mut offset = 0;
        for i in (0..indices.len()).rev() {
            let layout = &data.layouts[type_ids[i]];
            offset += layout.shape.offset(indices[i]) as i32;
        }

        self.locate_variable(data, parent_id).offset(offset)
    }

    fn create_stack_slot(&mut self, data: &Data, c: &mut FnContext, node_id: u32) -> StackSlot {
        let size = sizeof(data.types, data.type_id(node_id));
        c.b.create_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Location {
    base: LocationBase,
    offset: i32,
}

#[derive(Copy, Clone, Debug)]
enum LocationBase {
    Pointer(Value),
    Register(Variable),
    Stack(StackSlot),
}

impl Location {
    fn pointer(base_addr: Value, offset: i32) -> Self {
        let base = LocationBase::Pointer(base_addr);
        Self { base, offset }
    }
    fn stack(stack_slot: StackSlot, offset: i32) -> Self {
        let base = LocationBase::Stack(stack_slot);
        Self { base, offset }
    }
    fn get_addr(&self, c: &mut FnContext) -> Value {
        let ty = c.ptr_type;
        let ins = c.b.ins();
        match self.base {
            LocationBase::Pointer(base_addr) => {
                let offset = self.offset as i64;
                if offset == 0 {
                    base_addr
                } else {
                    ins.iadd_imm(base_addr, offset)
                }
            }
            LocationBase::Stack(stack_slot) => ins.stack_addr(ty, stack_slot, self.offset),
            LocationBase::Register(_) => unreachable!("cannot get address of register"),
        }
    }
    fn offset(self, extra_offset: i32) -> Self {
        Location {
            base: self.base,
            offset: self.offset + extra_offset,
        }
    }
    fn load_scalar(&self, c: &mut FnContext, flags: MemFlags) -> Value {
        let (ins, ty) = (c.b.ins(), c.ptr_type);
        match self.base {
            LocationBase::Pointer(base_addr) => ins.load(ty, flags, base_addr, self.offset),
            LocationBase::Register(variable) => c.b.use_var(variable),
            LocationBase::Stack(stack_slot) => ins.stack_load(ty, stack_slot, self.offset),
        }
    }
    fn load_value(&self, c: &mut FnContext, flags: MemFlags, layout: &Layout) -> Value {
        let (ins, ty) = (c.b.ins(), c.ptr_type);
        match self.base {
            LocationBase::Pointer(base_addr) => ins.load(ty, flags, base_addr, self.offset),
            LocationBase::Register(variable) => c.b.use_var(variable),
            LocationBase::Stack(stack_slot) => {
                if layout.size <= 8 {
                    ins.stack_load(ty, stack_slot, self.offset)
                } else {
                    ins.stack_addr(ty, stack_slot, self.offset)
                }
            }
        }
    }
    fn store(&self, c: &mut FnContext, value: Value, flags: MemFlags, layout: &Layout) {
        let (ins, ty) = (c.b.ins(), c.ptr_type);
        match self.base {
            // lvalue is a pointer, store in address, rvalue is scalar/aggregate
            LocationBase::Pointer(base_addr) => {
                ins.store(flags, value, base_addr, self.offset);
            }
            // lvalue is a register variable, rvalue must be a scalar
            LocationBase::Register(variable) => c.b.def_var(variable, value),
            // lvalue is a stack variable, rvalue is scalar/aggregate
            LocationBase::Stack(stack_slot) => {
                if layout.size <= 8 {
                    ins.stack_store(value, stack_slot, self.offset);
                } else {
                    let src_addr = value;
                    let dest_addr = ins.stack_addr(ty, stack_slot, self.offset);
                    c.b.emit_small_memory_copy(
                        c.target_config,
                        dest_addr,
                        src_addr,
                        layout.size as u64,
                        8,
                        8,
                        true,
                        MemFlags::new(),
                    );
                }
            }
        };
    }
    fn to_val(self, c: &mut FnContext, layout: &Layout) -> Val {
        let (ins, ty) = (c.b.ins(), c.ptr_type);
        match self.base {
            LocationBase::Register(variable) => Val::Scalar(c.b.use_var(variable)),
            LocationBase::Pointer(_) => Val::Reference(self, None),
            LocationBase::Stack(stack_slot) => {
                if layout.size <= 8 {
                    Val::Scalar(ins.stack_load(ty, stack_slot, self.offset))
                } else {
                    Val::Reference(self, None)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Val {
    Scalar(Value),
    Reference(Location, Option<Value>),
}

impl Val {
    fn cranelift_value(self, c: &mut FnContext, layout: &Layout) -> Value {
        match &self {
            Val::Scalar(value) => *value,
            // location is a pointer or stack slot
            Val::Reference(location, _) => {
                if layout.size <= 8 {
                    location.load_scalar(c, MemFlags::new())
                } else {
                    location.get_addr(c)
                }
            }
        }
    }
}

fn mangle_function_declaration(data: &Data, node_id: NodeId, includes_types: bool) -> String {
    let node = data.node(node_id);
    assert_eq!(node.tag, Tag::FunctionDecl);
    let mut full_name = data.tree.node_full_name(node_id);
    let lhs = data.node(node.lhs);
    let prototype = if lhs.tag == Tag::ParametricPrototype {
        data.node(lhs.rhs)
    } else {
        lhs
    };
    if includes_types {
        let parameters = data.node(prototype.lhs);
        if parameters.rhs > parameters.lhs {
            write!(full_name, "|");
        }
        for i in parameters.lhs..parameters.rhs {
            let ni = data.node_index(i);
            let ti = data.type_id(ni);
            write!(full_name, "{},", ti);
        }
    }
    full_name
}

fn mangle_function_call(data: &Data, node_id: NodeId) -> String {
    let node = data.node(node_id);
    assert_eq!(node.tag, Tag::Call);
    let function_expr = data.node(node.lhs);
    let base_name = data.tree.node_lexeme(node.lhs);
    let mut full_name = format!("{}", base_name);
    let arguments = data.node(node.rhs);
    if arguments.rhs > arguments.lhs {
        write!(full_name, "|");
    }
    for i in arguments.lhs..arguments.rhs {
        let ni = data.node_index(i);
        let ti = data.type_id(ni);
        write!(full_name, "{},", ti);
    }
    full_name
}
