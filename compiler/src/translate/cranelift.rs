use crate::translate::input::sizeof;
use core::mem;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

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
    pub fn new(input: &'a Input<'a>, output_name: String, use_jit: bool) -> Self {
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
                ("print_int", builtin::print_int as *const u8),
                ("alloc", builtin::alloc as *const u8),
                ("dealloc", builtin::dealloc as *const u8),
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
        let mut func_ids = Vec::new();
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
                        let func_name = self.data.tree.node_lexeme_offset(&node, -1);
                        let func_id = self.compile_function_decl(ni);
                        if func_name == "main" {
                            func_ids.push(func_id);
                            main_id = Some(func_id);
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
    pub fn compile_function_decl(&mut self, index: u32) -> FuncId {
        let node = self.data.node(index);
        assert_eq!(node.tag, Tag::FunctionDecl);
        let prototype = self.data.node(node.lhs);
        assert_eq!(prototype.tag, Tag::Prototype);
        let parameters = self.data.node(prototype.lhs);
        assert_eq!(parameters.tag, Tag::Parameters);
        let returns = self.data.node(prototype.rhs);

        let target_config = self.state.module.target_config();
        let ptr_type = target_config.pointer_type();
        Self::init_signature(&mut self.ctx.func, parameters, returns, ptr_type);
        let mut c = FnContext {
            b: FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx),
            ptr_type,
            target_config,
        };

        Self::compile_function_parameters(
            &mut self.state,
            &self.data,
            &mut c,
            ptr_type,
            prototype.lhs,
        );

        Self::compile_function_body(&mut self.state, &self.data, &mut c, node.rhs);

        c.b.finalize();
        let name = self.data.node_lexeme_offset(node, -1);
        println!("{} {}", name, c.b.func.display());
        let id = self
            .state
            .module
            .declare_function(name, Linkage::Export, &self.ctx.func.signature)
            .unwrap();
        self.state
            .module
            .define_function(id, &mut self.ctx)
            .unwrap();
        self.state.module.clear_context(&mut self.ctx);
        id
    }

    fn compile_function_parameters(
        state: &mut State,
        data: &Data,
        c: &mut FnContext,
        ptr_type: Type,
        node_id: NodeId,
    ) {
        let parameters = data.node(node_id);
        let entry_block = c.b.create_block();
        c.b.append_block_params_for_function_params(entry_block);
        c.b.switch_to_block(entry_block);
        c.b.seal_block(entry_block);

        // Define parameters as stack variables.
        let mut scalar_count = 0;
        for i in parameters.lhs..parameters.rhs {
            let ni = data.node_index(i);
            let stack_slot = state.create_stack_slot(&data, c, ni);
            let type_id = data.type_id(ni);
            let layout = &data.layouts[type_id];
            if layout.size <= 8 {
                let value = c.b.block_params(entry_block)[scalar_count];
                c.b.ins().stack_store(value, stack_slot, 0);
            } else {
                // Aggregate types are passed as pointers.
                // let addr = c.b.block_params(entry_block)[scalar_count];
                // for word_index in 0..(layout.size / 8) {
                //     let offset = word_index as i32 * 8;
                //     let value = c.b.ins().load(int, MemFlags::new(), addr, offset);
                //     c.b.ins().stack_store(value, var.slot, offset);
                // }
                let src_addr = c.b.block_params(entry_block)[scalar_count];
                let dest_addr = c.b.ins().stack_addr(ptr_type, stack_slot, 0);
                c.b.emit_small_memory_copy(
                    state.module.target_config(),
                    dest_addr,
                    src_addr,
                    layout.size as u64,
                    8,
                    8,
                    true,
                    MemFlags::new(),
                );
                // let size = c.b.ins().iconst(int, layout.size as i64);
                // c.b.call_memcpy(state.module.target_config(), dest_addr, src_addr, size);
            }
            scalar_count += 1;
            state.locations.insert(ni, Location::stack(stack_slot, 0));
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
                let condition_expr = self.compile_expr(data, c, node.lhs).value();
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
                    let condition_expr = self.compile_expr(data, c, if_nodes[i].lhs).value();
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
                    c.b.ins().jump(merge_block, &[]);
                }
                c.b.seal_block(merge_block);
                c.b.switch_to_block(merge_block);
            }
            Tag::VariableDecl => {
                // lhs: type
                // rhs: expr
                let slot = self.create_stack_slot(data, c, node_id);
                if node.rhs != 0 {
                    // Assume the init_expr is a scalar
                    let value = self.compile_expr(data, c, node.rhs).value();
                    c.b.ins().stack_store(value, slot, 0);
                }
                self.locations.insert(node_id, Location::stack(slot, 0));
            }
            Tag::Return => {
                let mut return_values = Vec::new();
                for i in node.lhs..node.rhs {
                    let ni = data.node_index(i);
                    let val = self.compile_expr(data, c, ni).value();
                    return_values.push(val);
                }
                c.b.ins().return_(&return_values[..]);
            }
            Tag::While => {
                let condition = self.compile_expr(data, c, node.lhs).value();
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
                let condition = self.compile_expr(data, c, node.lhs).value();
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
            Tag::Call => {
                let mut sig = self.module.make_signature();
                let function = data.node(node.lhs);
                let name = data.node_lexeme_offset(function, 0);

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
                    .declare_function(name, Linkage::Import, &sig)
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
        let lhs = self.compile_expr(data, c, node.lhs).value();
        let rhs = self.compile_expr(data, c, node.rhs).value();
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
        let def_id = data.get_definition_id(node_id);
        *self.locations.get(&def_id).expect("failed to get location")
    }

    fn locate_field(&self, data: &Data, node_id: NodeId) -> Location {
        let mut indices = Vec::new();
        let mut type_ids = Vec::new();
        let mut parent_id = node_id;
        let mut parent = data.node(parent_id);

        while parent.tag == Tag::Access {
            let field_index = data.get_definition_id(parent.rhs) as usize;
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

    pub fn value(self) -> Value {
        if let Val::Scalar(value) = self {
            return value;
        }
        panic!("expected scalar value, got aggregate")
    }
}
