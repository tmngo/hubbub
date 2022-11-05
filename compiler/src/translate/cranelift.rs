use crate::{
    analyze::{BuiltInFunction, Definition, Lookup},
    builtin,
    parse::{Node, NodeId, Tag},
    translate::input::{sizeof, Data, Input, Layout, Shape},
    typecheck::{BuiltInType, Type as Typ, TypeId, TypeIds},
    workspace::Workspace,
};
use cranelift::prelude::{
    codegen::{ir::StackSlot, Context},
    isa::{lookup, TargetFrontendConfig},
    settings,
    types::{F32, I32, I8},
    AbiParam, Block, FunctionBuilder, FunctionBuilderContext, InstBuilder, IntCC, MemFlags,
    Signature, StackSlotData, StackSlotKind, Type, Value, Variable,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use dlopen::raw::Library;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

pub trait CraneliftModule: Module {
    fn finalize(self: Box<Self>, id: FuncId, output_file: &Path) -> i64;
}

impl CraneliftModule for JITModule {
    fn finalize(mut self: Box<Self>, id: FuncId, _output_file: &Path) -> i64 {
        self.finalize_definitions();
        let main = self.get_finalized_function(id);
        let main: fn() -> i64 = unsafe { std::mem::transmute(main) };
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
    data_ctx: &'a mut DataContext,
    ptr_type: Type,
    target_config: TargetFrontendConfig,
}

pub struct Generator<'a> {
    pub builder_ctx: FunctionBuilderContext,
    pub ctx: Context,
    data_ctx: DataContext,
    data: Data<'a>,
    pub state: State,
}

impl<'a> Generator<'a> {
    pub fn new(
        workspace: &Workspace,
        input: &'a Input<'a>,
        output_name: String,
        use_jit: bool,
    ) -> Self {
        let flag_builder = settings::builder();
        // flag_builder.set("is_pic", "true").unwrap();
        let isa_builder = lookup(target_lexicon::HOST).unwrap();
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

            let dylib_paths = &workspace.library_files;

            // Leak to prevent the libraries from being dropped and unloaded.
            let dylibs = Box::leak(
                dylib_paths
                    .iter()
                    .map(|path| Library::open(path).unwrap())
                    .collect(),
            );
            let symbol_lookup_fn = Box::new(|name: &str| {
                for lib in &*dylibs {
                    if let Ok(symbol) = unsafe { lib.symbol::<*const u8>(name) } {
                        return Some(symbol);
                    }
                }
                None
            });
            jit_builder.symbol_lookup_fn(symbol_lookup_fn);

            Box::new(JITModule::new(jit_builder))
        } else {
            let object_builder =
                ObjectBuilder::new(isa, output_name, cranelift_module::default_libcall_names())
                    .unwrap();
            Box::new(ObjectModule::new(object_builder))
        };

        let layouts = input
            .types
            .iter()
            .enumerate()
            .map(|(i, typ)| Layout::new(input.types, typ, sizeof(input.types, i)))
            .collect();

        Self {
            builder_ctx: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_ctx: DataContext::new(),
            data: Data::new(input, layouts),
            state: State {
                module,
                locations: HashMap::new(),
                filled_blocks: HashSet::new(),
            },
        }
    }

    ///
    pub fn init_signature(
        data: &Data,
        signature: &mut Signature,
        parameters: &Node,
        returns: &Node,
        t: Type,
    ) {
        for i in parameters.lhs..parameters.rhs {
            // Tag::Field
            let ni = data.tree.node_index(i);
            let size = sizeof(data.types, data.type_id(ni));
            let t = if size == 1 { I8 } else { t };
            signature.params.push(AbiParam::new(t));
        }
        match returns.tag {
            Tag::Expressions => {
                for i in returns.lhs..returns.rhs {
                    let ni = data.tree.node_index(i);
                    let size = sizeof(data.types, data.type_id(ni));
                    let t = if size == 1 { I8 } else { t };
                    signature.returns.push(AbiParam::new(t));
                }
            }
            Tag::Identifier | Tag::Type => {
                signature.returns.push(AbiParam::new(t));
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
                if let Tag::FunctionDecl = node.tag {
                    // Skip generic functions with no specializations.
                    if self.data.node(node.lhs).lhs != 0
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
                        &mut self.data_ctx,
                        &mut self.state,
                        ni,
                    );
                    let fn_name = self.data.tree.name(ni);
                    if fn_name == "main" {
                        fn_ids.push(fn_id);
                        main_id = Some(fn_id);
                    }
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
        ctx: &mut Context,
        builder_ctx: &mut FunctionBuilderContext,
        data_ctx: &mut DataContext,
        state: &mut State,
        node_id: NodeId,
    ) -> FuncId {
        let target_config = state.module.target_config();
        let ptr_type = target_config.pointer_type();
        let mut c = FnContext {
            b: FunctionBuilder::new(&mut ctx.func, builder_ctx),
            data_ctx,
            ptr_type,
            target_config,
        };

        let node = data.node(node_id);
        c.b.func.signature = Self::compile_function_signature(state, data, &mut c, node.lhs);
        Self::compile_function_body(state, data, &mut c, node_id);
        c.b.finalize();
        let name = data.mangle_function_declaration(node_id, true);
        println!("{} :: {}", name, c.b.func.display());
        let fn_id = state
            .module
            .declare_function(&name, Linkage::Export, &c.b.func.signature)
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
    ) -> Signature {
        let mut signature = state.module.make_signature();
        let prototype = data.node(node_id);
        let parameters = data.node(data.tree.node_extra(prototype, 0));
        let returns = data.node(data.tree.node_extra(prototype, 1));
        Self::init_signature(data, &mut signature, parameters, returns, c.ptr_type);
        signature
    }

    fn compile_function_body(state: &mut State, data: &Data, c: &mut FnContext, node_id: NodeId) {
        let fn_decl = data.node(node_id);
        let prototype = data.node(fn_decl.lhs);
        let parameters = data.node(data.tree.node_extra(prototype, 0));
        let body = data.node(fn_decl.rhs);
        assert_eq!(fn_decl.tag, Tag::FunctionDecl);
        assert_eq!(body.tag, Tag::Block);

        let entry_block = c.b.create_block();
        c.b.append_block_params_for_function_params(entry_block);
        c.b.switch_to_block(entry_block);
        c.b.seal_block(entry_block);

        // Define parameters as stack variables.
        for i in parameters.lhs..parameters.rhs {
            let ni = data.node_index(i);
            let stack_slot = state.create_stack_slot(data, c, ni);
            let location = Location::stack(stack_slot, 0);
            state.locations.insert(ni, location);
            let parameter_index = (i - parameters.lhs) as usize;
            let value = c.b.block_params(entry_block)[parameter_index];
            let layout = data.layout(ni);
            location.store(c, value, MemFlags::new(), layout);
        }

        for i in body.lhs..body.rhs {
            let ni = data.node_index(i);
            state.compile_stmt(data, c, ni);
        }
    }
}

pub struct State {
    locations: HashMap<NodeId, Location>,
    pub module: Box<dyn CraneliftModule>,
    filled_blocks: HashSet<Block>,
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
                let rvalue = self.compile_expr_value(data, c, node.rhs);
                let flags = MemFlags::new();
                let lvalue = self.compile_lvalue(data, c, node.lhs);
                lvalue.store(c, rvalue, flags, layout);
            }
            Tag::If => {
                let condition_expr = self.compile_expr_value(data, c, node.lhs);
                let then_block = c.b.create_block();
                let merge_block = c.b.create_block();
                let body = data.node(node.rhs);
                c.b.ins().brz(condition_expr, merge_block, &[]);
                c.b.ins().jump(then_block, &[]);
                self.filled_blocks.insert(c.b.current_block().unwrap());
                c.b.seal_block(then_block);
                // then block
                c.b.switch_to_block(then_block);
                for i in body.lhs..body.rhs {
                    let index = data.node_index(i);
                    self.compile_stmt(data, c, index);
                }
                if !self.filled_blocks.contains(&c.b.current_block().unwrap()) {
                    c.b.ins().jump(merge_block, &[]);
                    self.filled_blocks.insert(c.b.current_block().unwrap());
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
                // If the last else-if block has no condition, it's an else.
                let has_else = if_nodes.last().unwrap().lhs == 0;
                let if_count = if has_else {
                    if_nodes.len() - 1
                } else {
                    if_nodes.len()
                };
                let merge_block = c.b.create_block();
                // Compile branches.
                for i in 0..if_count {
                    let condition_expr = self.compile_expr_value(data, c, if_nodes[i].lhs);
                    c.b.ins().brnz(condition_expr, then_blocks[i], &[]);
                    c.b.seal_block(then_blocks[i]);
                    if i < if_count - 1 {
                        // This is not the last else-if block.
                        let block = c.b.create_block();
                        c.b.ins().jump(block, &[]);
                        c.b.seal_block(block);
                        c.b.switch_to_block(block);
                    } else if !has_else {
                        // This is the last else-if block and there's no else.
                        c.b.ins().jump(merge_block, &[]);
                    } else {
                        // This is the last else-if block and there's an else.
                        c.b.ins().jump(then_blocks[if_count], &[]);
                        c.b.seal_block(then_blocks[if_count]);
                    }
                }
                // Compile block statements.
                for (i, if_node) in if_nodes.iter().enumerate() {
                    c.b.switch_to_block(then_blocks[i]);
                    let body = data.node(if_node.rhs);
                    for j in body.lhs..body.rhs {
                        let index = data.node_index(j);
                        self.compile_stmt(data, c, index);
                    }
                    if !self.filled_blocks.contains(&c.b.current_block().unwrap()) {
                        c.b.ins().jump(merge_block, &[]);
                    }
                }
                c.b.seal_block(merge_block);
                c.b.switch_to_block(merge_block);
            }
            Tag::VariableDecl => {
                // lhs: expressions
                // rhs: expressions
                let lhs = data.tree.node(node.lhs);
                assert_eq!(lhs.tag, Tag::Expressions);
                let mut locs = vec![];
                for i in lhs.lhs..lhs.rhs {
                    let ni = data.tree.node_index(i);
                    let slot = self.create_stack_slot(data, c, ni);
                    let location = Location::stack(slot, 0);
                    self.locations.insert(ni, location);
                    locs.push(location);
                }
                let rvalues_id = data.tree.node_extra(node, 1);
                if rvalues_id == 0 {
                    return;
                }
                let rhs = data.tree.node(rvalues_id);
                assert_eq!(rhs.tag, Tag::Expressions);
                let mut location_index = 0;
                if let Val::Multiple(values) = self.compile_expr(data, c, rvalues_id) {
                    let type_ids = data.type_ids(rvalues_id).all();
                    assert_eq!(type_ids.len(), values.len());
                    for (i, value) in values.iter().enumerate() {
                        let layout = &data.layouts[type_ids[i]];
                        locs[location_index].store(c, *value, MemFlags::new(), layout);
                        location_index += 1;
                    }
                } else {
                    unreachable!("rhs of variable declaration must be multiple-valued")
                }
            }
            Tag::Return => {
                let mut return_values = Vec::new();
                for i in node.lhs..node.rhs {
                    let ni = data.node_index(i);
                    let val = self.compile_expr_value(data, c, ni);
                    return_values.push(val);
                }
                c.b.ins().return_(&return_values[..]);
                self.filled_blocks.insert(c.b.current_block().unwrap());
            }
            Tag::While => {
                let condition = self.compile_expr_value(data, c, node.lhs);
                let while_block = c.b.create_block();
                let merge_block = c.b.create_block();
                // check condition
                // true? jump to loop body
                c.b.ins().brnz(condition, while_block, &[]);
                // false? jump to after loop
                c.b.ins().jump(merge_block, &[]);
                self.filled_blocks.insert(c.b.current_block().unwrap());
                // block_while:
                c.b.switch_to_block(while_block);
                let body = data.node(node.rhs);
                for i in body.lhs..body.rhs {
                    let ni = data.node_index(i);
                    self.compile_stmt(data, c, ni);
                }
                let condition = self.compile_expr_value(data, c, node.lhs);
                // brnz block_while
                c.b.ins().brnz(condition, while_block, &[]);
                c.b.seal_block(while_block);
                c.b.ins().jump(merge_block, &[]);
                c.b.seal_block(merge_block);
                self.filled_blocks.insert(c.b.current_block().unwrap());
                // block_merge:
                c.b.switch_to_block(merge_block);
            }
            _ => {
                self.compile_expr(data, c, node_id);
            }
        }
    }

    pub fn compile_expr_value(&mut self, data: &Data, c: &mut FnContext, node_id: NodeId) -> Value {
        self.compile_expr(data, c, node_id)
            .cranelift_value(data, c, node_id)
    }

    /// Returns a value. A value can be a scalar or an aggregate.
    /// NodeId -> Val
    pub fn compile_expr(&mut self, data: &Data, c: &mut FnContext, node_id: NodeId) -> Val {
        let ty = c.ptr_type;
        let node = data.node(node_id);
        match node.tag {
            Tag::Access => self.locate_field(data, node_id).to_val(data, c, node_id),
            Tag::Address => Val::Scalar(self.locate(data, node.lhs).get_addr(c)),
            Tag::Dereference => {
                let flags = MemFlags::new();
                let ptr_layout = data.layout(node.lhs);
                let ptr = self.locate(data, node.lhs).load_value(
                    c,
                    flags,
                    data.typ(node.lhs),
                    ptr_layout,
                );
                Location::pointer(ptr, 0).to_val(data, c, node_id)
            }
            Tag::Add | Tag::Mul => {
                let callee_id = node_id;
                self.compile_call(data, c, node_id, callee_id, true)
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
            Tag::Inequality => {
                let (lhs, rhs) = self.compile_children(data, c, node);
                Val::Scalar(c.b.ins().icmp(IntCC::NotEqual, lhs, rhs))
            }
            Tag::IntegerLiteral => self.compile_integer_literal(data, c, node_id, false),
            Tag::True => Val::Scalar(c.b.ins().iconst(ty, 1)),
            Tag::False => Val::Scalar(c.b.ins().iconst(ty, 0)),
            Tag::Call => {
                let callee_id = node.lhs;
                self.compile_call(data, c, node_id, callee_id, false)
            }
            Tag::Expressions => {
                let range = data.tree.range(node);
                let mut values = Vec::with_capacity(range.len());
                for i in range {
                    let ni = data.tree.node_index(i);
                    let val = self.compile_expr(data, c, ni);
                    match val {
                        Val::Multiple(xs) => values.extend(xs),
                        _ => values.push(val.cranelift_value(data, c, ni)),
                    }
                }
                Val::Multiple(values)
            }
            Tag::Identifier => self.locate(data, node_id).to_val(data, c, node_id),
            Tag::Negation => {
                if data.tree.node(node.lhs).tag == Tag::IntegerLiteral {
                    return self.compile_integer_literal(data, c, node.lhs, true);
                }
                let lhs = self.compile_expr_value(data, c, node.lhs);
                Val::Scalar(c.b.ins().ineg(lhs))
            }
            Tag::Subscript => {
                let lvalue = self.compile_lvalue(data, c, node_id);
                lvalue.to_val(data, c, node_id)
            }
            Tag::StringLiteral => {
                let string = data
                    .tree
                    .token_str(node.token)
                    .trim_matches('"')
                    .to_string();
                let length = string.len();

                // Create pointer to stored data.
                c.data_ctx.define(string.into_bytes().into_boxed_slice());
                let data_id = self.module.declare_anonymous_data(true, false).unwrap();
                self.module.define_data(data_id, c.data_ctx).unwrap();
                c.data_ctx.clear();
                let local_id = self.module.declare_data_in_func(data_id, c.b.func);
                let data_ptr = c.b.ins().symbol_value(ty, local_id);

                let slot = self.create_stack_slot(data, c, node_id);

                // Build string struct on stack.
                let length_value = c.b.ins().iconst(ty, length as i64);
                let layout = &Layout::new_scalar(8, 0);
                Location::stack(slot, 0).store(c, data_ptr, MemFlags::new(), layout);
                Location::stack(slot, 8).store(c, length_value, MemFlags::new(), layout);

                Location::stack(slot, 0).to_val(data, c, node_id)
            }
            _ => unreachable!("Invalid expression tag: {:?}", node.tag),
        }
    }

    fn compile_children(&mut self, data: &Data, c: &mut FnContext, node: &Node) -> (Value, Value) {
        let lhs = self.compile_expr_value(data, c, node.lhs);
        let rhs = self.compile_expr_value(data, c, node.rhs);
        (lhs, rhs)
    }

    fn compile_lvalue(&mut self, data: &Data, c: &mut FnContext, node_id: NodeId) -> Location {
        let node = data.tree.node(node_id);
        let flags = MemFlags::new();
        match node.tag {
            // a.x = b
            Tag::Access => self.locate_field(data, node_id),
            // a = b
            Tag::Identifier => self.locate_variable(data, node_id),
            // @a = b
            Tag::Dereference => {
                // Layout of referenced variable
                let ptr_layout = data.layout(node.lhs);
                let ptr = self.locate(data, node.lhs).load_value(
                    c,
                    flags,
                    data.typ(node.lhs),
                    ptr_layout,
                );
                Location::pointer(ptr, 0)
            }
            Tag::Subscript => {
                let arr_layout = data.layout(node.lhs);
                let stride = if let Shape::Array { stride, .. } = arr_layout.shape {
                    stride
                } else {
                    unreachable!();
                };
                let base = self.locate(data, node.lhs).load_value(
                    c,
                    flags,
                    data.typ(node.lhs),
                    arr_layout,
                );
                let index = self.compile_expr_value(data, c, node.rhs);
                dbg!(arr_layout);
                let offset = c.b.ins().imul_imm(index, stride as i64);
                let addr = c.b.ins().iadd(base, offset);
                // Offset for 1-based indexing.
                Location::pointer(addr, -(stride as i32))
            }
            _ => unreachable!("Invalid lvalue {:?} for assignment", node.tag),
        }
    }

    fn compile_call(
        &mut self,
        data: &Data,
        c: &mut FnContext,
        node_id: NodeId,
        callee_id: NodeId,
        binary: bool,
    ) -> Val {
        let ty = c.ptr_type;

        let definition = data.definitions.get(&callee_id).unwrap_or_else(|| {
            panic!("Definition not found: {}", "failed to get function decl id")
        });

        let name = match definition {
            Definition::BuiltInFunction(id) => {
                return self.compile_built_in_function(data, c, *id, node_id);
            }
            Definition::User(id) | Definition::Overload(id) => {
                data.mangle_function_declaration(*id, true)
            }
            Definition::Foreign(_) => data.tree.name(callee_id).to_string(),
            Definition::Resolved(id) => data.mangle_function_declaration(*id, true),
            Definition::BuiltIn(built_in_type) => {
                let node = data.tree.node(node_id);
                let args = data.node(node.rhs);
                let ni = data.node_index(args.lhs);
                let arg = self.compile_expr_value(data, c, ni);
                let ti = *built_in_type as TypeId;
                let from_type = cl_type(c.ptr_type, data.typ(ni));
                let to_typ = &data.types[ti];
                let to_type = cl_type(c.ptr_type, to_typ);
                return if to_type.bytes() < from_type.bytes() {
                    Val::Scalar(c.b.ins().ireduce(to_type, arg))
                } else if to_typ.is_signed() {
                    Val::Scalar(c.b.ins().sextend(to_type, arg))
                } else {
                    Val::Scalar(c.b.ins().uextend(to_type, arg))
                };
            }
            _ => unreachable!("Definition not found: {}", "failed to get function decl id"),
        };

        let node = data.tree.node(node_id);
        let args = if binary {
            vec![
                self.compile_expr_value(data, c, node.lhs),
                self.compile_expr_value(data, c, node.rhs),
            ]
        } else {
            data.tree
                .range(data.node(node.rhs))
                .map(|i| {
                    let ni = data.node_index(i);
                    self.compile_expr_value(data, c, ni)
                })
                .collect()
        };

        let mut sig = self.module.make_signature();
        sig.params = args.iter().map(|_| AbiParam::new(ty)).collect();

        // Assume one return value.
        let return_types = data.type_ids(node_id);
        let return_type = match return_types {
            TypeIds::Single(t) => {
                if *t != BuiltInType::Void as TypeId {
                    sig.returns.push(AbiParam::new(ty));
                }
                *t
            }
            TypeIds::Multiple(ts) => {
                for ti in ts {
                    let t = cl_type(c.ptr_type, &data.types[*ti]);
                    sig.returns.push(AbiParam::new(t));
                }
                return_types.first()
            }
        };

        let callee = self
            .module
            .declare_function(&name, Linkage::Import, &sig)
            .unwrap();
        let local_callee = self.module.declare_func_in_func(callee, c.b.func);
        let call = c.b.ins().call(local_callee, &args);
        if return_type != BuiltInType::Void as TypeId {
            let return_values = c.b.inst_results(call);
            if return_values.len() == 1 {
                Val::Scalar(return_values[0])
            } else {
                Val::Multiple(return_values.into())
            }
        } else {
            Val::Scalar(c.b.ins().iconst(ty, 0))
        }
    }

    fn compile_built_in_function(
        &mut self,
        data: &Data,
        c: &mut FnContext,
        built_in_function: BuiltInFunction,
        node_id: NodeId,
    ) -> Val {
        let node = data.tree.node(node_id);
        match built_in_function {
            BuiltInFunction::Add => {
                let a = self.compile_expr_value(data, c, node.lhs);
                let b = self.compile_expr_value(data, c, node.rhs);
                Val::Scalar(c.b.ins().iadd(a, b))
            }
            BuiltInFunction::Mul => {
                let a = self.compile_expr_value(data, c, node.lhs);
                let b = self.compile_expr_value(data, c, node.rhs);
                Val::Scalar(c.b.ins().imul(a, b))
            }
            BuiltInFunction::SizeOf => {
                let args = data.tree.node(node.rhs);
                let first_arg_id = data.tree.node_index(args.lhs);
                let type_id = data.type_id(first_arg_id);
                let value = sizeof(data.types, type_id) as i64;
                Val::Scalar(c.b.ins().iconst(c.ptr_type, value))
            }
        }
    }

    fn compile_integer_literal(
        &mut self,
        data: &Data,
        c: &mut FnContext,
        node_id: NodeId,
        negative: bool,
    ) -> Val {
        // Doesn't use cl_type, so we can default to ptr_type.
        let ty = match data.typ(node_id) {
            Typ::Unsigned8 => I8,
            Typ::Unsigned32 => I32,
            _ => c.ptr_type,
        };
        let token_str = data.tree.node_lexeme(node_id);
        let value = token_str.parse::<i64>().unwrap();
        Val::Scalar(c.b.ins().iconst(ty, if negative { -value } else { value }))
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
        c.b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size))
    }
}

pub fn cl_type(ptr_type: Type, typ: &Typ) -> Type {
    match typ {
        Typ::Integer => ptr_type,
        Typ::Unsigned32 => I32,
        Typ::Unsigned8 => I8,
        Typ::Float => F32,
        Typ::Boolean => I8,
        Typ::Pointer { .. } => ptr_type,
        _ => unreachable!("Invalid type: {typ:?} is not a primitive Cranelift type."),
    }
}

// https://github.com/bjorn3/rustc_codegen_cranelift/blob/master/src/pointer.rs
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
    fn load_scalar(&self, c: &mut FnContext, flags: MemFlags, typ: &Typ) -> Value {
        let ins = c.b.ins();
        let ty = cl_type(c.ptr_type, typ);
        match self.base {
            LocationBase::Pointer(base_addr) => ins.load(ty, flags, base_addr, self.offset),
            LocationBase::Register(variable) => c.b.use_var(variable),
            LocationBase::Stack(stack_slot) => ins.stack_load(ty, stack_slot, self.offset),
        }
    }
    fn load_value(&self, c: &mut FnContext, flags: MemFlags, typ: &Typ, layout: &Layout) -> Value {
        let (ins, ty) = (c.b.ins(), c.ptr_type);
        match self.base {
            LocationBase::Pointer(_) | LocationBase::Register(_) => self.load_scalar(c, flags, typ),
            LocationBase::Stack(stack_slot) => {
                if layout.size <= 8 {
                    self.load_scalar(c, flags, typ)
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
    fn to_val(self, data: &Data, c: &mut FnContext, node_id: NodeId) -> Val {
        let ins = c.b.ins();
        let typ = data.typ(node_id);
        let layout = data.layout(node_id);
        match self.base {
            LocationBase::Register(variable) => Val::Scalar(c.b.use_var(variable)),
            LocationBase::Pointer(_) => Val::Reference(self, None),
            LocationBase::Stack(stack_slot) => {
                if layout.size <= 8 {
                    let ty = cl_type(c.ptr_type, typ);
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
    Multiple(Vec<Value>),
}

impl Val {
    fn cranelift_value(self, data: &Data, c: &mut FnContext, node_id: NodeId) -> Value {
        let typ = data.typ(node_id);
        let layout = data.layout(node_id);
        match &self {
            Val::Scalar(value) => *value,
            // location is a pointer or stack slot
            Val::Reference(location, _) => {
                if layout.size <= 8 {
                    location.load_scalar(c, MemFlags::new(), typ)
                } else {
                    location.get_addr(c)
                }
            }
            _ => unreachable!("cannot get cranelift_value for multiple"),
        }
    }
}
