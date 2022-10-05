use crate::{
    analyze::{BuiltInFunction, Definition, Lookup},
    parse::{Node, NodeId, Tag},
    translate::input::{Data, Input, Layout},
    typecheck::{BuiltInType, Type as Typ, TypeId},
};
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::ExecutionEngine,
    module::Module,
    passes::PassManager,
    targets::{CodeModel, FileType, RelocMode, Target, TargetMachine, TargetTriple},
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, IntType},
    values::{
        BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue,
    },
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

pub fn compile(input: &Input, use_jit: bool, obj_filename: &Path) {
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
    println!("{}", codegen.module.print_to_string().to_string_lossy());

    // Write object file.
    target_machine
        .write_to_file(&codegen.module, FileType::Object, obj_filename)
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
                    // let is_overloaded = self.data.definitions.contains_key(&ni);
                    let name = self.data.mangle_function_declaration(ni, true);
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
        None
    }

    pub fn compile_function_decl(
        &self,
        state: &mut State<'ctx>,
        node_id: NodeId,
    ) -> FunctionValue<'ctx> {
        let node = self.data.node(node_id);
        // let is_overloaded = self.data.definitions.contains_key(&node_id);
        let name = self.data.mangle_function_declaration(node_id, true);

        let fn_value = self.module.get_function(&name).unwrap();

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
            _ => {
                if returns.rhs - returns.lhs == 1 {
                    llvm_type(self.context, data.types, data.type_id(prototype.rhs))
                } else {
                    self.context.struct_type(&[], false).into()
                }
            }
        };
        let fn_type = ret_type.fn_type(argslice, false);
        let fn_value = self.module.add_function(name, fn_type, None);

        println!("{}", crate::format_red!("{:?}", fn_value.get_name()));
        dbg!(fn_type);

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
            let stack_addr = builder.build_alloca(llvm_type, "alloca_param");
            let location = Location::new(stack_addr, 0);
            state.locations.insert(ni, location);
            let parameter_index = i - parameters.lhs;
            let value = fn_value.get_nth_param(parameter_index).unwrap();
            self.builder.build_store(stack_addr, value);
        }
    }

    fn compile_function_body(&self, state: &mut State<'ctx>, node_id: NodeId) {
        let data = &self.data;
        let body = data.node(node_id);
        for i in body.lhs..body.rhs {
            let ni = data.node_index(i);
            self.compile_stmt(state, ni);
        }
        if self
            .builder
            .get_insert_block()
            .unwrap()
            .get_terminator()
            .is_none()
        {
            let ret_type = state
                .function
                .unwrap()
                .get_type()
                .get_return_type()
                .unwrap();
            self.builder.build_return(Some(&ret_type.const_zero()));
        }
    }

    pub fn compile_stmt(&self, state: &mut State<'ctx>, node_id: NodeId) {
        let (data, builder) = (&self.data, &self.builder);
        let node = data.node(node_id);
        match node.tag {
            Tag::Assign => {
                assert_eq!(data.type_id(node.lhs), data.type_id(node.rhs));
                let rvalue = self.compile_expr(state, node.rhs);
                let lvalue = self.compile_lvalue(state, node.lhs);
                let lvalue = self.builder.build_pointer_cast(
                    lvalue,
                    rvalue.get_type().ptr_type(AddressSpace::Generic),
                    "ptr_cast",
                );
                self.builder.build_store(lvalue, rvalue);
            }
            Tag::If => {
                let parent_fn = state.function.unwrap();
                let condition_expr = self.compile_expr(state, node.lhs).into_int_value();
                let then_block = self.context.append_basic_block(parent_fn, "then");
                let merge_block = self.context.append_basic_block(parent_fn, "merge");
                let body = data.node(node.rhs);

                builder.build_conditional_branch(condition_expr, then_block, merge_block);
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
                    assert_eq!(if_node.tag, Tag::If);
                    if_nodes.push(if_node);
                    then_blocks.push(self.context.append_basic_block(parent_fn, "then"));
                }
                // If the last else-if block has no condition, it's an else.
                let has_else = if_nodes.last().unwrap().lhs == 0;
                let if_count = if has_else {
                    if_nodes.len() - 1
                } else {
                    if_nodes.len()
                };
                let merge_block = self.context.append_basic_block(parent_fn, "merge");
                // Compile branches.
                for i in 0..if_count {
                    let condition_expr = self.compile_expr(state, if_nodes[i].lhs).into_int_value();
                    if i < if_count - 1 {
                        // This is not the last else-if block.
                        let block = self.context.append_basic_block(parent_fn, "block");
                        builder.build_conditional_branch(condition_expr, then_blocks[i], block);
                        builder.position_at_end(block);
                    } else if !has_else {
                        // This is the last else-if block and there's no else.
                        builder.build_conditional_branch(
                            condition_expr,
                            then_blocks[i],
                            merge_block,
                        );
                    } else {
                        // This is the last else-if block and there's an else.
                        builder.build_conditional_branch(
                            condition_expr,
                            then_blocks[i],
                            then_blocks[if_count],
                        );
                    }
                }
                // Compile block statements.
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
                let lhs = data.tree.node(node.token);
                match lhs.tag {
                    Tag::Expressions => {
                        let mut locs = vec![];
                        for i in lhs.lhs..lhs.rhs {
                            let ni = data.tree.node_index(i);
                            let type_id = data.type_id(ni);
                            let llvm_type = llvm_type(self.context, data.types, type_id);
                            let stack_addr = builder.build_alloca(llvm_type, "alloca_local");
                            let location = Location::new(stack_addr, 0);
                            state.locations.insert(ni, location);
                            locs.push(location);
                        }
                        let rhs = data.tree.node(node.rhs);
                        if rhs.tag == Tag::Expressions {
                            for i in rhs.lhs..rhs.rhs {
                                let ni = data.tree.node_index(i);
                                let value = self.compile_expr(state, ni);
                                self.builder
                                    .build_store(locs[(i - rhs.lhs) as usize].base, value);
                            }
                        }
                    }
                    Tag::Identifier => {
                        let ni = node.token;
                        let type_id = data.type_id(ni);
                        let llvm_type = llvm_type(self.context, data.types, type_id);
                        let stack_addr = builder.build_alloca(llvm_type, "alloca_local");
                        let location = Location::new(stack_addr, 0);
                        state.locations.insert(ni, location);
                        if node.rhs != 0 {
                            let value = self.compile_expr(state, node.rhs);
                            self.builder.build_store(location.base, value);
                        }
                    }
                    _ => {}
                }
            }
            Tag::Return => {
                // let mut return_values = Vec::new();
                // for i in node.lhs..node.rhs {
                //     let ni = data.node_index(i);
                //     let val = self.compile_expr(state, ni);
                //     let zero = self.context.i64_type().const_zero();
                //     dbg!(val, zero);
                //     return_values.push(zero);
                // }
                if node.lhs == node.rhs {
                    println!("returning none");
                    let unit_value = self.context.const_struct(&[], false).as_basic_value_enum();
                    builder.build_return(Some(&unit_value));
                } else {
                    println!("returning some");
                    let val = self.compile_expr(state, data.node_index(node.lhs));
                    builder.build_return(Some(dbg!(&val)));
                }
            }
            Tag::While => {
                let parent_fn = state.function.unwrap();
                let condition_expr = self.compile_expr(state, node.lhs).into_int_value();
                let while_block = self.context.append_basic_block(parent_fn, "while_block");
                let merge_block = self.context.append_basic_block(parent_fn, "merge_block");
                // check condition
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
                builder.build_conditional_branch(condition_expr, while_block, merge_block);
                // block_merge:
                builder.position_at_end(merge_block);
            }
            _ => {
                self.compile_expr(state, node_id);
            }
        }
    }

    pub fn compile_expr(&self, state: &mut State<'ctx>, node_id: NodeId) -> BasicValueEnum<'ctx> {
        let (data, builder) = (&self.data, &self.builder);
        let node = data.node(node_id);
        match node.tag {
            Tag::Access => {
                let container = self.compile_expr(state, node.lhs);
                let field_id = data
                    .definitions
                    .get_definition_id(node_id, "failed to lookup field definition");
                let field = data.node(field_id);
                let field_index = data.node_index(field.rhs + 1);
                match container {
                    BasicValueEnum::PointerValue(pointer) => {
                        let gep = builder.build_struct_gep(pointer, field_index, "").unwrap();
                        self.builder.build_load(gep, "")
                    }
                    BasicValueEnum::StructValue(value) => {
                        builder.build_extract_value(value, field_index, "").unwrap()
                    }
                    _ => unreachable!("cannot gep_struct for non-struct value"),
                }
            }
            Tag::Address => self.locate(state, node.lhs).base.into(),
            Tag::Dereference => {
                let location = self.compile_lvalue(state, node.lhs);
                let ptr = builder
                    .build_load(location, "pointer_value")
                    .into_pointer_value();
                builder.build_load(ptr, "deref")
            }
            Tag::Add => {
                let definition = data.definitions.get(&node_id).unwrap_or_else(|| {
                    panic!("Definition not found: {}", "failed to get function decl id")
                });
                let args = vec![
                    self.compile_expr(state, node.lhs),
                    self.compile_expr(state, node.rhs),
                ];
                self.compile_call(node_id, definition, args)
            }
            Tag::BitwiseShiftL => {
                let (lhs, rhs) = self.compile_children(state, node);
                builder.build_left_shift(lhs, rhs, "left_shift").into()
            }
            Tag::BitwiseShiftR => {
                let (lhs, rhs) = self.compile_children(state, node);
                builder
                    .build_right_shift(lhs, rhs, true, "right_shift")
                    .into()
            }
            Tag::BitwiseXor => {
                let (lhs, rhs) = self.compile_children(state, node);
                builder.build_xor(lhs, rhs, "xor").into()
            }
            Tag::Sub => {
                let (lhs, rhs) = self.compile_children(state, node);
                builder.build_int_sub(lhs, rhs, "int_sub").into()
            }
            Tag::Div => {
                let (lhs, rhs) = self.compile_children(state, node);
                builder
                    .build_int_signed_div(lhs, rhs, "int_signed_div")
                    .into()
            }
            Tag::Mul => {
                let (lhs, rhs) = self.compile_children(state, node);
                builder.build_int_mul(lhs, rhs, "int_mul").into()
            }
            Tag::Equality => {
                let (lhs, rhs) = self.compile_children(state, node);
                builder
                    .build_int_compare(IntPredicate::EQ, lhs, rhs, "int_compare_eq")
                    .into()
            }
            Tag::Greater => {
                let (lhs, rhs) = self.compile_children(state, node);
                builder
                    .build_int_compare(IntPredicate::SGT, lhs, rhs, "int_compare_sgt")
                    .into()
            }
            Tag::Less => {
                let (lhs, rhs) = self.compile_children(state, node);
                builder
                    .build_int_compare(IntPredicate::SLT, lhs, rhs, "int_compare_slt")
                    .into()
            }
            Tag::Grouping => self.compile_expr(state, node.lhs),
            Tag::IntegerLiteral => {
                let token_str = data.tree.node_lexeme(node_id);
                let value = token_str.parse::<i64>().unwrap();
                let type_id = data.type_id(node_id);
                let typ = &data.types[type_id as usize];
                let llvm_type = llvm_type(self.context, data.types, type_id);
                llvm_type
                    .into_int_type()
                    .const_int(value as u64, typ.is_signed())
                    .into()
                // self.context
                //     .i64_type()
                //     .const_int(value as u64, false)
                //     .into()
            }
            Tag::True => self.context.bool_type().const_int(1, false).into(),
            Tag::False => self.context.bool_type().const_int(0, false).into(),
            Tag::Call => {
                let definition = data.definitions.get(&node.lhs).unwrap_or_else(|| {
                    panic!("Definition not found: {}", "failed to get function decl id")
                });
                let args: Vec<BasicValueEnum> = data
                    .tree
                    .range(data.node(node.rhs))
                    .map(|i| self.compile_expr(state, data.node_index(i)))
                    .collect();
                self.compile_call(node_id, definition, args)
            }
            Tag::Identifier => {
                let name = data.tree.name(node_id);
                let location = self.locate(state, node_id);
                builder.build_load(location.base, name)
            }
            Tag::Subscript => {
                let lvalue = self.compile_lvalue(state, node_id);
                builder.build_load(lvalue, "subscript")
            }
            _ => unreachable!("Invalid expression tag: {:?}", node.tag),
        }
    }

    fn compile_lvalue(&self, state: &mut State<'ctx>, node_id: NodeId) -> PointerValue<'ctx> {
        let (data, builder) = (&self.data, &self.builder);
        let node = data.node(node_id);
        match node.tag {
            Tag::Access => {
                let field_id = data
                    .definitions
                    .get_definition_id(node_id, "failed to lookup field definition");
                let field = data.node(field_id);
                let field_index = data.node_index(field.rhs + 1);
                let struct_ptr = self.compile_lvalue(state, node.lhs);
                builder
                    .build_struct_gep(struct_ptr, field_index, "")
                    .unwrap()
            }
            // a = b
            Tag::Identifier => self.locate(state, node_id).base,
            // @a = b
            Tag::Dereference => {
                let location = self.compile_lvalue(state, node.lhs);
                builder
                    .build_load(location, "pointer_value")
                    .into_pointer_value()
            }
            Tag::Subscript => {
                let array_ptr = self.compile_lvalue(state, node.lhs);
                let index_value = self.compile_expr(state, node.rhs).into_int_value();
                let zero = self.context.i64_type().const_int(0, false);
                unsafe { builder.build_gep(array_ptr, &[zero, index_value], "gep") }
            }
            _ => unreachable!("Invalid lvalue {:?} for assignment", node.tag),
        }
    }

    fn compile_children(
        &self,
        state: &mut State<'ctx>,
        node: &Node,
    ) -> (IntValue<'ctx>, IntValue<'ctx>) {
        let lhs = self.compile_expr(state, node.lhs).into_int_value();
        let rhs = self.compile_expr(state, node.rhs).into_int_value();
        (lhs, rhs)
    }

    fn compile_call(
        &self,
        node_id: NodeId,
        definition: &Definition,
        args: Vec<BasicValueEnum<'ctx>>,
    ) -> BasicValueEnum<'ctx> {
        let (data, builder) = (&self.data, &self.builder);
        let name = match &definition {
            Definition::BuiltInFunction(id) => {
                return self.compile_built_in_function(*id, args);
            }
            Definition::User(id) | Definition::Foreign(id) | Definition::Overload(id) => {
                data.mangle_function_declaration(*id, true)
            }
            Definition::Resolved(id) => data.mangle_function_declaration(*id, true),
            _ => unreachable!("Definition not found: {}", "failed to get function decl id"),
        };

        let callee = self
            .module
            .get_function(&name)
            .unwrap_or_else(|| panic!("failed to get module function \"{}\"", name));

        // Assume one return value.
        let return_type = data.type_id(node_id);

        let mut argiter = args.iter();
        let argslice = argiter.by_ref();
        let argslice = argslice
            .map(|&val| val.into())
            .collect::<Vec<BasicMetadataValueEnum>>();
        let argslice = argslice.as_slice();
        let call_site_value = builder.build_call(callee, argslice, &name);
        if return_type != BuiltInType::Void as TypeId {
            call_site_value
                .try_as_basic_value()
                .left()
                .expect("basic value expected")
        } else {
            self.context.const_struct(&[], false).as_basic_value_enum()
        }
    }

    fn compile_built_in_function(
        &self,
        built_in_function: BuiltInFunction,
        args: Vec<BasicValueEnum<'ctx>>,
    ) -> BasicValueEnum<'ctx> {
        match built_in_function {
            BuiltInFunction::Add => self
                .builder
                .build_int_add(
                    args[0].into_int_value(),
                    args[1].into_int_value(),
                    "int_add",
                )
                .into(),
        }
    }

    fn locate(&self, state: &mut State<'ctx>, node_id: NodeId) -> Location<'ctx> {
        let node = self.data.node(node_id);
        match node.tag {
            Tag::Identifier => {
                let def_id = self
                    .data
                    .definitions
                    .get_definition_id(node_id, "failed to look up variable definition");
                *state
                    .locations
                    .get(&def_id)
                    .expect("failed to get identifier location")
            }
            _ => unreachable!("Cannot locate node with tag {:?}", node.tag),
        }
    }

    // fn gep_struct(&self, value: BasicValueEnum<'ctx>, field_index: u32) -> BasicValueEnum<'ctx> {
    //     match value {
    //         BasicValueEnum::PointerValue(pointer) => {
    //             let gep = self
    //                 .builder
    //                 .build_struct_gep(pointer, field_index, "")
    //                 .unwrap();
    //             self.builder.build_load(gep, "")
    //         }
    //         BasicValueEnum::StructValue(value) => self
    //             .builder
    //             .build_extract_value(value, field_index, "")
    //             .unwrap(),
    //         _ => unreachable!("cannot gep_struct for non-struct value"),
    //     }
    // }
}

pub fn llvm_type<'ctx>(
    context: &'ctx Context,
    types: &Vec<Typ>,
    type_id: usize,
) -> BasicTypeEnum<'ctx> {
    match &types[type_id] {
        Typ::Array { typ, length, .. } => llvm_type(context, types, *typ)
            .array_type(*length as u32)
            .into(),
        Typ::Pointer { typ, .. } => llvm_type(context, types, *typ)
            .ptr_type(AddressSpace::Generic)
            .into(),
        Typ::Struct { fields, .. } => {
            let mut field_types = Vec::with_capacity(fields.len());
            for id in fields {
                field_types.push(llvm_type(context, types, *id))
            }
            context.struct_type(&field_types, false).into()
        }
        Typ::Void => context.struct_type(&[], false).into(),
        Typ::Integer => context.i64_type().into(),
        Typ::Unsigned8 => context.i8_type().into(),
        Typ::Boolean => context.bool_type().into(),
        _ => unreachable!("Invalid type"),
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Location<'ctx> {
    base: PointerValue<'ctx>,
    offset: i32,
}

impl<'ctx> Location<'ctx> {
    fn new(base: PointerValue<'ctx>, offset: i32) -> Self {
        Self { base, offset }
    }
    // fn offset(self, extra_offset: i32) -> Self {
    //     Location {
    //         base: self.base,
    //         offset: self.offset + extra_offset,
    //     }
    // }
}
