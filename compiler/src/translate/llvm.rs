use crate::{
    analyze::Lookup,
    parse::{Node, NodeId, Tag},
    translate::input::{Data, Input, Layout},
    typecheck::{Type as Typ, TypeId, TypeIndex},
};
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::ExecutionEngine,
    module::Module,
    passes::PassManager,
    targets::{CodeModel, FileType, RelocMode, Target, TargetMachine, TargetTriple},
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, IntType},
    values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue, IntValue, PointerValue},
    AddressSpace, IntPredicate, OptimizationLevel,
};
use std::{collections::HashMap, path::Path};

struct Generator<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    ptr_sized_int_type: IntType<'ctx>,
    pass_manager: PassManager<FunctionValue<'ctx>>,
    data: Data<'ctx>,
}

struct State<'a> {
    locations: HashMap<NodeId, Location<'a>>,
    function: Option<FunctionValue<'a>>,
}

pub fn compile(input: &Input, use_jit: bool, obj_filename: &str) {
    let context = Context::create();
    let module = context.create_module("");
    let execution_engine = if use_jit {
        module
            .create_jit_execution_engine(OptimizationLevel::Default)
            .unwrap()
    } else {
        module.create_execution_engine().unwrap()
    };
    let builder = context.create_builder();
    let layouts = input
        .types
        .iter()
        .map(|typ| Layout::new(input.types, typ, 8))
        .collect();
    let data = Data::new(input, layouts);
    let target_data = execution_engine.get_target_data();
    let ptr_sized_int_type = context.ptr_sized_int_type(target_data, None);

    let pass_manager = PassManager::create(&module);

    pass_manager.add_instruction_combining_pass();
    pass_manager.add_reassociate_pass();
    pass_manager.add_gvn_pass();
    pass_manager.add_cfg_simplification_pass();
    pass_manager.add_basic_alias_analysis_pass();
    pass_manager.add_promote_memory_to_register_pass();
    pass_manager.add_instruction_combining_pass();
    pass_manager.add_reassociate_pass();

    pass_manager.initialize();

    let codegen = Generator {
        context: &context,
        module,
        builder,
        execution_engine,
        data,
        pass_manager,
        ptr_sized_int_type,
    };

    let mut state = State {
        locations: HashMap::new(),
        function: None,
    };
    codegen.compile_nodes(&mut state);

    let triple = TargetTriple::create(&target_lexicon::HOST.to_string());
    let target = Target::from_triple(&triple).unwrap();
    let host_cpu = TargetMachine::get_host_cpu_name();
    let features = TargetMachine::get_host_cpu_features();

    let target_machine = target
        .create_target_machine(
            &triple,
            host_cpu.to_str().unwrap(),
            features.to_str().unwrap(),
            OptimizationLevel::Default,
            RelocMode::Default,
            CodeModel::Default,
        )
        .unwrap();
    codegen
        .module
        .set_data_layout(&target_machine.get_target_data().get_data_layout());
    codegen.module.set_triple(&target_machine.get_triple());

    // Print LLVM IR.
    codegen.module.print_to_stderr();

    // Write object file.
    target_machine
        .write_to_file(&codegen.module, FileType::Object, Path::new(&obj_filename))
        .unwrap();
}

impl<'ctx> Generator<'ctx> {
    pub fn compile_nodes(&self, state: &mut State<'ctx>) -> Option<i64> {
        let root = self.data.tree.node(0);
        let mut fn_values = Vec::new();
        let mut main_fn = None;

        for i in root.lhs..root.rhs {
            let module_index = self.data.tree.node_index(i);
            let module = self.data.tree.node(module_index);
            if module.tag != Tag::Module {
                continue;
            }
            for i in module.lhs..module.rhs {
                let ni = self.data.tree.node_index(i);
                let node = self.data.tree.node(ni);
                if let Tag::FunctionDecl = node.tag {
                    // Skip generic functions with no specializations.
                    if self.data.tree.node(node.lhs).tag == Tag::ParametricPrototype
                        && !self.data.type_parameters.contains_key(&ni)
                    {
                        continue;
                    }
                    let name = self.data.mangle_function_declaration(ni, false);
                    self.compile_function_signature(node.lhs, &name);
                };
            }
        }

        for i in root.lhs..root.rhs {
            let module_index = self.data.tree.node_index(i);
            let module = self.data.tree.node(module_index);

            if module.tag != Tag::Module {
                continue;
            }

            for i in module.lhs..module.rhs {
                let ni = self.data.tree.node_index(i);
                let node = self.data.tree.node(ni);
                // println!("{:?}", node.tag);
                println!("compiling {}", self.data.tree.name(ni));
                if let Tag::FunctionDecl = node.tag {
                    // Skip generic functions with no specializations.
                    if self.data.tree.node(node.lhs).tag == Tag::ParametricPrototype
                        && !self.data.type_parameters.contains_key(&ni)
                    {
                        continue;
                    }
                    // Skip function signatures
                    if node.rhs == 0 {
                        continue;
                    }
                    let fn_value = self.compile_function_decl(state, ni);
                    let fn_name = self.data.tree.name(ni);
                    if fn_name == "main" {
                        fn_values.push(fn_value);
                        main_fn = Some(fn_value);
                    }
                };
            }
        }
        if main_fn.is_some() {
            // return Some(self.state.module.finalize(id, filename));
        }
        println!("compile_nodes");

        None
    }

    pub fn compile_function_decl(
        &self,
        state: &mut State<'ctx>,
        node_id: NodeId,
    ) -> FunctionValue<'ctx> {
        let node = self.data.node(node_id);
        let name = self.data.mangle_function_declaration(node_id, false);

        let fn_value = self.module.get_function(&name).unwrap();
        // let fn_value = self.compile_function_signature(state, node.lhs, &name);

        let entry_block = self.context.append_basic_block(fn_value, "entry");
        self.builder.position_at_end(entry_block);
        let prototype = self.data.node(node.lhs);

        self.compile_function_parameters(state, prototype.lhs, fn_value);
        state.function = Some(fn_value);
        self.compile_function_body(state, node.rhs);

        if fn_value.verify(true) {
            self.pass_manager.run_on(&fn_value);
        } else {
            unsafe {
                fn_value.delete();
            }
        }
        fn_value
    }

    fn compile_function_signature(&self, node_id: NodeId, name: &str) -> FunctionValue<'ctx> {
        let data = &self.data;
        let prototype = data.node(node_id);
        let parameters = data.node(prototype.lhs);
        let returns = data.node(prototype.rhs);

        let mut arg_types = vec![];
        for i in parameters.lhs..parameters.rhs {
            let ni = data.node_index(i);
            let type_id = data.type_id(ni);
            arg_types.push(llvm_type(self.context, data.types, type_id))
        }

        let mut argiter = arg_types.iter();
        let argslice = argiter.by_ref();
        let argslice = argslice
            .map(|&val| val.into())
            .collect::<Vec<BasicMetadataTypeEnum>>();
        let argslice = argslice.as_slice();

        let ret_type: BasicTypeEnum = match returns.tag {
            Tag::Expressions => {
                let mut return_types = vec![];
                for i in returns.lhs..returns.rhs {
                    let ni = data.node_index(i);
                    let type_id = data.type_id(ni);
                    return_types.push(llvm_type(self.context, data.types, type_id));
                }
                self.context.struct_type(&return_types, false).into()
            }
            Tag::Identifier => llvm_type(self.context, data.types, data.type_id(prototype.rhs)),
            _ => self.context.struct_type(&[], false).into(),
        };
        let fn_type = ret_type.fn_type(argslice, false);
        let fn_value = self.module.add_function(name, fn_type, None);

        fn_value
    }

    fn compile_function_parameters(
        &self,
        state: &mut State<'ctx>,
        node_id: NodeId,
        fn_value: FunctionValue<'ctx>,
    ) {
        let (data, builder) = (&self.data, &self.builder);
        let parameters = data.node(node_id);

        // Define parameters as stack variables.
        for i in parameters.lhs..parameters.rhs {
            let ni = data.node_index(i);
            let type_id = data.type_id(ni);
            let llvm_type = llvm_type(self.context, data.types, type_id);
            let stack_addr = builder.build_alloca(llvm_type, "alloca");
            let location = Location::stack(stack_addr, 0);
            state.locations.insert(ni, location);
            let parameter_index = i - parameters.lhs;
            let value = fn_value.get_nth_param(parameter_index).unwrap();
            let layout = data.layout(ni);
            location.store(self, value, layout);
        }
    }

    fn compile_function_body(&self, state: &mut State<'ctx>, node_id: NodeId) {
        let data = &self.data;
        let body = data.node(node_id);
        for i in body.lhs..body.rhs {
            let ni = data.node_index(i);
            self.compile_stmt(state, ni);
        }
    }

    pub fn compile_stmt(&self, state: &mut State<'ctx>, node_id: NodeId) {
        let (data, builder) = (&self.data, &self.builder);
        let node = data.node(node_id);
        match node.tag {
            Tag::Assign => {
                // lhs: expr
                // rhs: expr
                assert_eq!(data.type_id(node.lhs), data.type_id(node.rhs));
                let layout = data.layout(node.rhs);
                let left_node = data.node(node.lhs);
                let right_value = self.compile_expr(state, node.rhs);
                let left_location = match left_node.tag {
                    // a.x = b
                    Tag::Access => self.locate_field(state, node.lhs),
                    // a = b
                    Tag::Identifier => self.locate_variable(state, node.lhs),
                    // @a = b
                    Tag::Dereference => {
                        // Layout of referenced variable
                        let ptr_layout = data.layout(left_node.lhs);
                        let location = self.locate(state, left_node.lhs);
                        let ptr = location
                            .load_value(self, ptr_layout)
                            .into_pointer_value()
                            .clone();
                        Location::pointer(ptr.clone(), 0)
                    }
                    Tag::Subscript => {
                        let arr_layout = data.layout(left_node.lhs);
                        let stride = if let Shape::Array { stride, .. } = arr_layout.shape {
                            stride
                        } else {
                            unreachable!();
                        };
                        let location = self.locate(state, left_node.lhs);
                        let base = location.load_value(self, arr_layout).into_pointer_value();
                        let index_value = self.compile_expr(state, node.rhs).into_int_value();
                        let stride_value = self.context.i64_type().const_int(stride as u64, false);
                        let offset_value: IntValue =
                            builder.build_int_mul(index_value, stride_value.into(), "offset");
                        let addr = unsafe { builder.build_gep(base, &[offset_value], "gep") };
                        Location::pointer(addr, 0)
                    }
                    _ => unreachable!("Invalid lvalue {:?} for assignment", left_node.tag),
                };
                left_location.store(self, right_value, layout);
            }
            Tag::If => {
                let parent_fn = state.function.unwrap();
                let condition_expr = self.compile_expr(state, node.lhs);
                let then_block = self.context.append_basic_block(parent_fn, "then");
                let merge_block = self.context.append_basic_block(parent_fn, "merge");
                let body = data.node(node.rhs);

                // Branch if zero
                let zero = self.ptr_sized_int_type.const_zero();
                let condition_expr = builder.build_int_compare(
                    IntPredicate::EQ,
                    condition_expr.into_int_value(),
                    zero,
                    "ifcond",
                );
                builder.build_conditional_branch(condition_expr, then_block, merge_block);
                builder.build_unconditional_branch(then_block);
                // then block
                builder.position_at_end(then_block);
                for i in body.lhs..body.rhs {
                    let ni = data.node_index(i);
                    self.compile_stmt(state, ni);
                }
                if builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_none()
                {
                    builder.build_unconditional_branch(merge_block);
                }
                // merge block
                builder.position_at_end(merge_block);
            }
            Tag::IfElse => {
                let parent_fn = state.function.unwrap();
                let mut if_nodes = Vec::new();
                let mut then_blocks = Vec::new();
                for i in node.lhs..node.rhs {
                    let index = data.node_index(i);
                    let if_node = data.node(index);
                    if_nodes.push(if_node);
                    then_blocks.push(self.context.append_basic_block(parent_fn, "then"));
                }
                let if_count = if if_nodes.last().unwrap().lhs == 0 {
                    if_nodes.len() - 1
                } else {
                    if_nodes.len()
                };
                let merge_block = self.context.append_basic_block(parent_fn, "merge");
                for i in 0..if_count {
                    let condition_expr = self.compile_expr(state, if_nodes[i].lhs);
                    let zero = self.ptr_sized_int_type.const_zero();
                    let condition_expr = builder.build_int_compare(
                        IntPredicate::NE,
                        condition_expr.into_int_value(),
                        zero,
                        "ifcond",
                    );
                    if i < if_count - 1 {
                        let block = self.context.append_basic_block(parent_fn, "block");
                        builder.build_conditional_branch(condition_expr, then_blocks[i], block);
                        builder.position_at_end(block);
                    }
                    if i == if_nodes.len() - 1 {
                        builder.build_conditional_branch(
                            condition_expr,
                            then_blocks[i],
                            merge_block,
                        );
                    }
                }
                for i in if_count..if_nodes.len() {
                    builder.build_unconditional_branch(then_blocks[i]);
                }
                for (i, if_node) in if_nodes.iter().enumerate() {
                    builder.position_at_end(then_blocks[i]);
                    let body = data.node(if_node.rhs);
                    for j in body.lhs..body.rhs {
                        let index = data.node_index(j);
                        self.compile_stmt(state, index);
                    }
                    if builder
                        .get_insert_block()
                        .unwrap()
                        .get_terminator()
                        .is_none()
                    {
                        builder.build_unconditional_branch(merge_block);
                    }
                }
                builder.position_at_end(merge_block);
            }
            Tag::VariableDecl => {
                // lhs: type
                // rhs: expr
                let type_id = data.type_id(node_id);
                let llvm_type = llvm_type(self.context, data.types, type_id);
                let stack_addr = builder.build_alloca(llvm_type, "alloca");

                let location = Location::stack(stack_addr, 0);
                state.locations.insert(node_id, location);
                if node.rhs != 0 {
                    let layout = data.layout(node.rhs);
                    let value = self.compile_expr(state, node.rhs);
                    location.store(self, value, layout);
                }
            }
            Tag::Return => {
                let mut return_values = Vec::new();
                for i in node.lhs..node.rhs {
                    let ni = data.node_index(i);
                    let val = self.compile_expr(state, ni);
                    return_values.push(val);
                }
                if return_values.len() == 0 {
                    builder.build_return(None);
                } else {
                    builder.build_return(Some(&return_values[0]));
                }
            }
            Tag::While => {
                let parent_fn = state.function.unwrap();
                let condition_expr = self.compile_expr(state, node.lhs).into_int_value();
                let while_block = self.context.append_basic_block(parent_fn, "while_block");
                let merge_block = self.context.append_basic_block(parent_fn, "merge_block");
                // check condition
                // let zero = self.context.bool_type().const_zero();
                // let condition_expr = builder
                //     .build_int_compare(
                //         IntPredicate::NE,
                //         condition_expr.into_int_value(),
                //         zero,
                //         "whilecond",
                //     )
                //     .into();
                // true? jump to loop body
                // false? jump to after loop
                builder.build_conditional_branch(condition_expr, while_block, merge_block);
                // block_while:
                builder.position_at_end(while_block);
                let body = data.node(node.rhs);
                for i in body.lhs..body.rhs {
                    let ni = data.node_index(i);
                    self.compile_stmt(state, ni);
                }
                let condition_expr = self.compile_expr(state, node.lhs).into_int_value();
                // brnz block_while
                // let zero = self.context.bool_type().const_zero();
                // let condition_expr = builder
                //     .build_int_compare(
                //         IntPredicate::NE,
                //         condition_expr.into_int_value(),
                //         zero,
                //         "whilecond",
                //     )
                //     .into();
                builder.build_conditional_branch(condition_expr, while_block, merge_block);
                // block_merge:
                builder.position_at_end(merge_block);
            }
            _ => {
                self.compile_expr(state, node_id);
            }
        }
    }

    pub fn compile_expr(&self, state: &mut State<'ctx>, node_id: NodeId) -> BasicValueEnum {
        let (data, builder) = (&self.data, &self.builder);
        let node = data.node(node_id);
        let layout = data.layout(node_id);
        match node.tag {
            Tag::Access => self.locate_field(state, node_id).load_value(self, layout),
            Tag::Address => self
                .locate(state, node.lhs)
                .get_addr(self.context, builder)
                .into(),
            Tag::Dereference => {
                let ptr_layout = data.layout(node.lhs);
                let ptr = self
                    .locate(state, node.lhs)
                    .load_value(self, ptr_layout)
                    .into_pointer_value()
                    .clone();
                Location::pointer(ptr, 0).load_value(self, layout)
            }
            Tag::Add => {
                let (lhs, rhs) = self.compile_children(state, node);
                BasicValueEnum::IntValue(builder.build_int_add(lhs, rhs, "int_add"))
            }
            Tag::BitwiseShiftL => {
                let (lhs, rhs) = self.compile_children(state, node);
                BasicValueEnum::IntValue(builder.build_left_shift(lhs, rhs, "left_shift"))
            }
            Tag::BitwiseShiftR => {
                let (lhs, rhs) = self.compile_children(state, node);
                BasicValueEnum::IntValue(builder.build_right_shift(lhs, rhs, false, "right_shift"))
            }
            Tag::BitwiseXor => {
                let (lhs, rhs) = self.compile_children(state, node);
                BasicValueEnum::IntValue(builder.build_xor(lhs, rhs, "xor"))
            }
            Tag::Sub => {
                let (lhs, rhs) = self.compile_children(state, node);
                BasicValueEnum::IntValue(builder.build_int_sub(lhs, rhs, "int_sub"))
            }
            Tag::Div => {
                let (lhs, rhs) = self.compile_children(state, node);
                BasicValueEnum::IntValue(builder.build_int_signed_div(lhs, rhs, "int_signed_div"))
            }
            Tag::Mul => {
                let (lhs, rhs) = self.compile_children(state, node);
                BasicValueEnum::IntValue(builder.build_int_mul(lhs, rhs, "int_mul"))
            }
            Tag::Equality => {
                let (lhs, rhs) = self.compile_children(state, node);
                BasicValueEnum::IntValue(builder.build_int_compare(
                    IntPredicate::EQ,
                    lhs,
                    rhs,
                    "int_compare_eq",
                ))
            }
            Tag::Greater => {
                let (lhs, rhs) = self.compile_children(state, node);
                BasicValueEnum::IntValue(builder.build_int_compare(
                    IntPredicate::SGT,
                    lhs,
                    rhs,
                    "int_compare_sgt",
                ))
            }
            Tag::Less => {
                let (lhs, rhs) = self.compile_children(state, node);
                BasicValueEnum::IntValue(builder.build_int_compare(
                    IntPredicate::SLT,
                    lhs,
                    rhs,
                    "int_compare_slt",
                ))
            }
            Tag::Grouping => self.compile_expr(state, node.lhs),
            Tag::IntegerLiteral => {
                let token_str = data.tree.node_lexeme(node_id);
                let value = token_str.parse::<i64>().unwrap();
                BasicValueEnum::IntValue(self.context.i64_type().const_int(value as u64, false))
            }
            Tag::True => BasicValueEnum::IntValue(self.context.i64_type().const_int(1, false)),
            Tag::False => BasicValueEnum::IntValue(self.context.i64_type().const_int(0, false)),
            Tag::Call => {
                let function_id = data
                    .definitions
                    .get_definition_id(node.lhs, "failed to get function decl");

                println!("name:         {}", data.tree.name(node.lhs));
                println!(
                    "mangled call: {}",
                    data.mangle_function_declaration(function_id, false)
                );

                let name = data.mangle_function_declaration(function_id, false);
                let callee = self.module.get_function(&name).unwrap();

                // Arguments
                let arguments = data.node(node.rhs);
                let mut args = Vec::new();

                for i in arguments.lhs..arguments.rhs {
                    let ni = data.node_index(i);
                    let value = self.compile_expr(state, ni);
                    args.push(value);
                }

                // Assume one return value.
                let return_type = data.type_id(node_id);

                let mut argiter = args.iter();
                let argslice = argiter.by_ref();
                let argslice = argslice
                    .map(|&val| val.into())
                    .collect::<Vec<BasicMetadataValueEnum>>();
                let argslice = argslice.as_slice();

                let call_site_value = builder.build_call(callee, argslice, &name);
                if return_type != TypeIndex::Void as TypeId {
                    call_site_value.try_as_basic_value().left().unwrap()
                } else {
                    BasicValueEnum::IntValue(self.context.i64_type().const_int(0, false))
                }
            }
            Tag::Identifier => self.locate(state, node_id).load_value(self, layout),
            Tag::Subscript => {
                let arr_layout = data.layout(node.lhs);
                let stride = if let Shape::Array { stride, .. } = arr_layout.shape {
                    stride
                } else {
                    unreachable!();
                };
                let base = self
                    .locate(state, node.lhs)
                    .load_value(self, arr_layout)
                    .into_pointer_value();
                dbg!(arr_layout);
                let index_value = self.compile_expr(state, node.rhs).into_int_value();
                let stride_value = self.context.i64_type().const_int(stride as u64, false);
                let offset_value = builder
                    .build_int_mul(index_value, stride_value.into(), "offset")
                    .into();
                let addr = unsafe { builder.build_gep(base, &[offset_value], "gep") };
                let location = Location::pointer(addr, 0);
                location.load_value(self, layout)
            }
            _ => unreachable!("Invalid expression tag: {:?}", node.tag),
        }
    }

    fn compile_children(&self, state: &mut State<'ctx>, node: &Node) -> (IntValue, IntValue) {
        let lhs = if let BasicValueEnum::IntValue(value) = self.compile_expr(state, node.lhs) {
            value
        } else {
            unreachable!()
        };
        let rhs = if let BasicValueEnum::IntValue(value) = self.compile_expr(state, node.rhs) {
            value
        } else {
            unreachable!()
        };
        (lhs, rhs)
    }

    fn locate(&self, state: &mut State<'ctx>, node_id: NodeId) -> Location {
        let node = self.data.node(node_id);
        match node.tag {
            Tag::Access => self.locate_field(state, node_id),
            Tag::Identifier => self.locate_variable(state, node_id),
            _ => unreachable!("Cannot locate node with tag {:?}", node.tag),
        }
    }

    fn locate_variable(&self, state: &mut State<'ctx>, node_id: NodeId) -> Location {
        let def_id = self
            .data
            .definitions
            .get_definition_id(node_id, "failed to look up variable definition");
        *state
            .locations
            .get(&def_id)
            .expect("failed to get identifier location")
    }

    fn locate_field(&self, state: &mut State<'ctx>, node_id: NodeId) -> Location {
        let data = &self.data;
        let mut indices = Vec::new();
        let mut type_ids = Vec::new();
        let mut parent_id = node_id;
        let mut parent = data.node(parent_id);

        while parent.tag == Tag::Access {
            // let field_id = data
            //     .definitions
            //     .get_definition_id(parent_id, "failed to lookup field definition");
            // let field = data.node(field_id);
            // let source_index = data.node_index(field.rhs + 1);
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

        self.locate_variable(state, parent_id).offset(offset)
    }
}

pub fn llvm_type<'ctx>(
    context: &'ctx Context,
    types: &Vec<Typ>,
    type_id: usize,
) -> BasicTypeEnum<'ctx> {
    dbg!(&types[type_id]);
    match &types[type_id] {
        Typ::Array { typ, length } => llvm_type(context, types, *typ)
            .array_type(*length as u32)
            .into(),
        Typ::Pointer { typ } => llvm_type(context, types, *typ)
            .ptr_type(AddressSpace::Generic)
            .into(),
        Typ::Struct { fields } => {
            let mut field_types = Vec::with_capacity(fields.len());
            for id in fields {
                field_types.push(llvm_type(context, types, *id))
            }
            context.struct_type(&field_types, false).into()
        }
        Typ::Void => context.struct_type(&[], false).into(),
        Typ::Integer => context.i64_type().into(),
        _ => unreachable!("Invalid type"),
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Location<'ctx> {
    base: LocationBase<'ctx>,
    offset: i32,
}

#[derive(Copy, Clone, Debug)]
enum LocationBase<'ctx> {
    Pointer(PointerValue<'ctx>),
    Register(BasicValueEnum<'ctx>),
    Stack(PointerValue<'ctx>),
}

impl<'ctx> Location<'ctx> {
    fn pointer(base_addr: PointerValue<'ctx>, offset: i32) -> Self {
        let base = LocationBase::Pointer(base_addr);
        Self { base, offset }
    }
    fn stack(stack_addr: PointerValue<'ctx>, offset: i32) -> Self {
        let base = LocationBase::Stack(stack_addr);
        Self { base, offset }
    }
    fn get_addr(&self, context: &'ctx Context, builder: &'ctx Builder) -> PointerValue<'ctx> {
        match self.base {
            LocationBase::Pointer(base_addr) => {
                if self.offset == 0 {
                    base_addr
                } else {
                    let offset = context.i64_type().const_int(self.offset as u64, false);
                    unsafe { builder.build_gep(base_addr, &[offset], "gep") }
                }
            }
            LocationBase::Stack(stack_addr) => {
                let offset = context.i64_type().const_int(self.offset as u64, false);
                unsafe { builder.build_gep(stack_addr, &[offset], "gep") }
            }
            LocationBase::Register(_) => unreachable!("cannot get address of register"),
        }
    }
    fn offset(self, extra_offset: i32) -> Self {
        Location {
            base: self.base,
            offset: self.offset + extra_offset,
        }
    }
    fn load_scalar(self, generator: &'ctx Generator) -> BasicValueEnum<'ctx> {
        let (context, builder) = (generator.context, &generator.builder);
        match self.base {
            LocationBase::Pointer(base_addr) => {
                if self.offset == 0 {
                    builder.build_load(base_addr, "load")
                } else {
                    let offset = context.i64_type().const_int(self.offset as u64, false);
                    let ptr = unsafe { builder.build_gep(base_addr, &[offset], "gep") };
                    builder.build_load(ptr, "load")
                }
            }
            LocationBase::Register(value) => value,
            LocationBase::Stack(stack_addr) => {
                if self.offset == 0 {
                    builder.build_load(stack_addr, "load")
                } else {
                    let offset = context.i64_type().const_int(self.offset as u64, false);
                    let ptr = unsafe { builder.build_gep(stack_addr, &[offset], "gep") };
                    builder.build_load(ptr, "load")
                }
            }
        }
    }
    fn load_value(self, generator: &'ctx Generator, layout: &Layout) -> BasicValueEnum<'ctx> {
        let (context, builder) = (&generator.context, &generator.builder);
        match self.base {
            LocationBase::Pointer(_) | LocationBase::Register(_) => self.load_scalar(generator),
            LocationBase::Stack(stack_addr) => {
                if layout.size <= 8 {
                    self.load_scalar(generator)
                } else {
                    let offset = context.i64_type().const_int(self.offset as u64, false);
                    let ptr = unsafe { builder.build_gep(stack_addr, &[offset], "gep") };
                    BasicValueEnum::PointerValue(ptr)
                }
            }
        }
    }
    fn store(&self, generator: &Generator, value: BasicValueEnum, layout: &Layout) {
        let (context, builder, ptr_sized_int_type) = (
            &generator.context,
            &generator.builder,
            generator.ptr_sized_int_type,
        );
        match self.base {
            // lvalue is a pointer, store in address, rvalue is scalar/aggregate
            LocationBase::Pointer(base_addr) => {
                let ptr = if self.offset == 0 {
                    base_addr
                } else {
                    let offset = context.i64_type().const_int(self.offset as u64, false);
                    unsafe { builder.build_gep(base_addr, &[offset], "gep") }
                };
                builder.build_store(ptr, value);
            }
            // lvalue is a register variable, rvalue must be a scalar
            LocationBase::Register(_) => unreachable!("Cannot store in register"),
            // lvalue is a stack variable, rvalue is scalar/aggregate
            LocationBase::Stack(stack_addr) => {
                if layout.size <= 8 {
                    let ptr = if self.offset == 0 {
                        stack_addr
                    } else {
                        let offset = context.i64_type().const_int(self.offset as u64, false);
                        unsafe { builder.build_gep(stack_addr, &[offset], "gep") }
                    };
                    builder.build_store(ptr, value);
                } else {
                    let src_addr = value.into_pointer_value();
                    let offset = context.i64_type().const_int(self.offset as u64, false);
                    let dest_addr = unsafe { builder.build_gep(stack_addr, &[offset], "gep") };
                    let size = ptr_sized_int_type.const_int(layout.size as u64, false);
                    builder
                        .build_memcpy(dest_addr, layout.align, src_addr, layout.align, size)
                        .ok();
                }
            }
        };
    }

    // fn to_value(self, generator: &Generator, layout: &Layout) -> BasicValueEnum {
    //     match self.base {
    //         LocationBase::Register(value) => value,
    //         LocationBase::Pointer(_) => Val::Reference(self, None),
    //         LocationBase::Stack(stack_slot) => {
    //             if layout.size <= 8 {
    //                 Val::Scalar(ins.stack_load(ty, stack_slot, self.offset))
    //             } else {
    //                 Val::Reference(self, None)
    //             }
    //         }
    //     }
    // }
}
