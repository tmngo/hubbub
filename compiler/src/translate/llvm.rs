use crate::{
    analyze::{BuiltInFunction, Definition, Lookup},
    parse::{Node, NodeId, Tag},
    translate::input::{sizeof, Data, Input, Layout},
    types::{Type as Typ, TypeId, T},
};
use codespan_reporting::diagnostic::Diagnostic;
use inkwell::{
    basic_block::BasicBlock,
    builder::{Builder, BuilderError},
    context::Context,
    execution_engine::ExecutionEngine,
    module::Module,
    passes::PassManager,
    targets::{CodeModel, FileType, RelocMode, Target, TargetMachine, TargetTriple},
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, IntType},
    values::{
        BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue,
        StructValue,
    },
    AddressSpace, IntPredicate, OptimizationLevel,
};
use std::{collections::HashMap, path::Path};

#[allow(dead_code)]
struct Generator<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    b: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    ptr_sized_int_type: IntType<'ctx>,
    pass_manager: PassManager<FunctionValue<'ctx>>,
    data: Data<'ctx>,
    state: State<'ctx>,
}

struct State<'a> {
    locations: HashMap<NodeId, Location<'a>>,
    function: Option<FunctionValue<'a>>,
    block_expr_stack: Vec<BasicBlock<'a>>,
    block_yields: Vec<Vec<(BasicValueEnum<'a>, BasicBlock<'a>)>>,
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
        .enumerate()
        .map(|(t, _)| Layout::new(input.types, t))
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

    let mut codegen = Generator {
        context: &context,
        module,
        b: builder,
        execution_engine,
        data,
        pass_manager,
        ptr_sized_int_type,
        state: State {
            locations: HashMap::new(),
            function: None,
            block_expr_stack: Vec::new(),
            block_yields: Vec::new(),
        },
    };
    codegen.compile_nodes();

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
    pub fn compile_nodes(&mut self) -> Option<i64> {
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
                    if self.data.tree.node(node.lhs).lhs != 0 {
                        if !self.data.type_parameters.contains_key(&ni) {
                            continue;
                        }
                        for type_arguments in self.data.type_parameters.get(&ni).unwrap().values() {
                            self.data.active_type_parameters.push(type_arguments);
                            let name = self.data.mangle_function_declaration(
                                ni,
                                true,
                                type_arguments,
                                None,
                            );
                            self.compile_function_signature(node.lhs, &name);
                            self.data.active_type_parameters.pop();
                        }
                        continue;
                    }
                    // let is_overloaded = self.data.definitions.contains_key(&ni);
                    let name = self.data.mangle_function_declaration(ni, true, &[], None);
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
                    if self.data.tree.node(node.lhs).lhs != 0 {
                        if !self.data.type_parameters.contains_key(&ni) {
                            continue;
                        }
                        for type_arguments in self.data.type_parameters.get(&ni).unwrap().values() {
                            self.data.active_type_parameters.push(type_arguments);
                            let name = self.data.mangle_function_declaration(
                                ni,
                                true,
                                type_arguments,
                                None,
                            );
                            if let Err(err) = self.compile_function_decl(ni, name.clone()) {
                                panic!("Error compiling function declaration '{}': {}", name, err)
                            }
                            self.data.active_type_parameters.pop();
                        }
                        continue;
                    }
                    // Skip function signatures
                    if node.rhs == 0 {
                        continue;
                    }
                    let name = self.data.mangle_function_declaration(ni, true, &[], None);
                    let fn_value = match self.compile_function_decl(ni, name.clone()) {
                        Ok(value) => value,
                        Err(err) => {
                            panic!("Error compiling function declaration '{}': {}", name, err)
                        }
                    };
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
        &mut self,
        node_id: NodeId,
        name: String,
    ) -> Result<FunctionValue<'ctx>, BuilderError> {
        let node = *self.data.node(node_id);
        // let is_overloaded = self.data.definitions.contains_key(&node_id);

        let fn_value = self.module.get_function(&name).unwrap();

        let entry_block = self.context.append_basic_block(fn_value, "entry");
        self.b.position_at_end(entry_block);
        let prototype = self.data.node(node.lhs);

        let parameters_id = self.data.tree.node_extra(prototype, 0);
        self.compile_function_parameters(parameters_id, fn_value)?;
        self.state.function = Some(fn_value);
        self.compile_function_body(node.rhs)?;

        if fn_value.verify(true) {
            self.pass_manager.run_on(&fn_value);
        } else {
            unsafe {
                fn_value.delete();
            }
        }
        Ok(fn_value)
    }

    fn compile_function_signature(&self, node_id: NodeId, name: &str) -> FunctionValue<'ctx> {
        let data = &self.data;
        let prototype = data.node(node_id);
        let parameters = data.node(self.data.tree.node_extra(prototype, 0));
        let returns_id = self.data.tree.node_extra(prototype, 1);
        let returns = data.node(returns_id);

        let mut arg_types = vec![];
        for i in parameters.lhs..parameters.rhs {
            let ni = data.node_index(i);
            let type_id = data.type_id(ni);
            arg_types.push(llvm_type(self.context, data, type_id))
        }

        let mut argiter = arg_types.iter();
        let argslice = argiter.by_ref();
        let argslice = argslice
            .map(|&val| val.into())
            .collect::<Vec<BasicMetadataTypeEnum>>();
        let argslice = argslice.as_slice();

        let ret_type = if returns_id == 0 {
            self.context.struct_type(&[], false).into()
        } else {
            assert_eq!(returns.tag, Tag::Expressions);
            let mut return_types = vec![];
            for i in returns.lhs..returns.rhs {
                let ni = data.node_index(i);
                let type_id = data.type_id(ni);
                return_types.push(llvm_type(self.context, data, type_id));
            }
            if return_types.len() == 1 {
                return_types[0]
            } else {
                self.context.struct_type(&return_types, false).into()
            }
        };
        let fn_type = ret_type.fn_type(argslice, false);
        let fn_value = self.module.add_function(name, fn_type, None);

        // println!("{}", crate::format_red!("{:?}", fn_value.get_name()));
        // dbg!(fn_type);

        fn_value
    }

    fn compile_function_parameters(
        &mut self,
        node_id: NodeId,
        fn_value: FunctionValue<'ctx>,
    ) -> Result<(), BuilderError> {
        let (data, builder) = (&self.data, &self.b);
        let parameters = data.node(node_id);

        // Define parameters as stack variables.
        for i in parameters.lhs..parameters.rhs {
            let ni = data.node_index(i);
            let type_id = data.type_id(ni);
            let llvm_type = llvm_type(self.context, data, type_id);
            let stack_addr = builder.build_alloca(llvm_type, "alloca_param")?;
            let location = Location::new(stack_addr, 0);
            self.state.locations.insert(ni, location);
            let parameter_index = i - parameters.lhs;
            let value = fn_value.get_nth_param(parameter_index).unwrap();
            self.b.build_store(stack_addr, value)?;
        }
        Ok(())
    }

    fn compile_function_body(&mut self, node_id: NodeId) -> Result<(), BuilderError> {
        let body = self.data.node(node_id);
        for i in body.lhs..body.rhs {
            let ni = self.data.node_index(i);
            self.compile_stmt(ni)?;
        }
        if self
            .b
            .get_insert_block()
            .unwrap()
            .get_terminator()
            .is_none()
        {
            let ret_type = self
                .state
                .function
                .unwrap()
                .get_type()
                .get_return_type()
                .unwrap();
            self.b.build_return(Some(&ret_type.const_zero()))?;
        }
        Ok(())
    }

    pub fn compile_stmt(&mut self, node_id: NodeId) -> Result<(), BuilderError> {
        let node = *self.data.node(node_id);
        match node.tag {
            Tag::Assign => {
                let rvalue = self.compile_expr(node.rhs)?;
                let lvalue = self.compile_lvalue(node.lhs)?;
                let lvalue = self.b.build_pointer_cast(
                    lvalue,
                    rvalue.get_type().ptr_type(AddressSpace::default()),
                    "ptr_cast",
                )?;
                self.b.build_store(lvalue, rvalue)?;
            }
            Tag::If => {
                let parent_fn = self.state.function.unwrap();
                let condition_expr = self.compile_expr(node.lhs)?.into_int_value();
                let then_block = self.context.append_basic_block(parent_fn, "then");
                let merge_block = self.context.append_basic_block(parent_fn, "merge");
                let body = self.data.node(node.rhs);

                self.b
                    .build_conditional_branch(condition_expr, then_block, merge_block)?;
                // then block
                self.b.position_at_end(then_block);
                for i in body.lhs..body.rhs {
                    let ni = self.data.node_index(i);
                    self.compile_stmt(ni)?;
                }
                if self
                    .b
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_none()
                {
                    self.b.build_unconditional_branch(merge_block)?;
                }
                // merge block
                self.b.position_at_end(merge_block);
            }
            Tag::IfElse => {
                let parent_fn = self.state.function.unwrap();
                let mut if_nodes = Vec::new();
                let mut then_blocks = Vec::new();
                for i in node.lhs..node.rhs {
                    let index = self.data.node_index(i);
                    let if_node = *self.data.node(index);
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
                    let condition_expr = self.compile_expr(if_nodes[i].lhs)?.into_int_value();
                    if i < if_count - 1 {
                        // This is not the last else-if block.
                        let block = self.context.append_basic_block(parent_fn, "block");
                        self.b
                            .build_conditional_branch(condition_expr, then_blocks[i], block)?;
                        self.b.position_at_end(block);
                    } else if !has_else {
                        // This is the last else-if block and there's no else.
                        self.b.build_conditional_branch(
                            condition_expr,
                            then_blocks[i],
                            merge_block,
                        )?;
                    } else {
                        // This is the last else-if block and there's an else.
                        self.b.build_conditional_branch(
                            condition_expr,
                            then_blocks[i],
                            then_blocks[if_count],
                        )?;
                    }
                }
                // Compile block statements.
                for (i, if_node) in if_nodes.iter().enumerate() {
                    self.b.position_at_end(then_blocks[i]);
                    let body = self.data.node(if_node.rhs);
                    for j in body.lhs..body.rhs {
                        let index = self.data.node_index(j);
                        self.compile_stmt(index)?;
                    }
                    if self
                        .b
                        .get_insert_block()
                        .unwrap()
                        .get_terminator()
                        .is_none()
                    {
                        self.b.build_unconditional_branch(merge_block)?;
                    }
                }
                self.b.position_at_end(merge_block);
            }
            Tag::VariableDecl => {
                // lhs: expressions
                // rhs: expressions
                let identifiers = self.data.tree.node(node.lhs);
                let mut locs = vec![];
                for i in self.data.tree.range(identifiers) {
                    let ni = self.data.tree.node_index(i);
                    let type_id = self.data.type_id(ni);
                    let llvm_type = llvm_type(self.context, &self.data, type_id);
                    let stack_addr = self.b.build_alloca(llvm_type, "alloca_local").unwrap();
                    let location = Location::new(stack_addr, 0);
                    self.state.locations.insert(ni, location);
                    locs.push(location);
                }
                let rvalues_id = self.data.tree.node_extra(&node, 1);
                if rvalues_id == 0 {
                    return Ok(());
                }
                let _rhs = self.data.tree.node(rvalues_id);
                for (i, value) in self.compile_exprs(rvalues_id)?.iter().enumerate() {
                    self.b.build_store(locs[i].base, *value)?;
                }
            }
            Tag::Return => {
                if node.lhs == 0 {
                    let unit_value = self.context.const_struct(&[], false).as_basic_value_enum();
                    self.b.build_return(Some(&unit_value))?;
                } else {
                    let mut return_values = Vec::new();
                    let expressions = self.data.tree.node(node.lhs);
                    for i in self.data.tree.range(expressions) {
                        let val = self.compile_expr(self.data.node_index(i))?;
                        return_values.push(val);
                    }
                    if return_values.len() == 1 {
                        self.b.build_return(Some(&return_values[0]))?;
                    } else {
                        self.b.build_aggregate_return(&return_values)?;
                    };
                }
            }
            Tag::Yield => {
                let values = if node.lhs == 0 {
                    Vec::new()
                } else {
                    let expressions = self.data.tree.node(node.lhs);
                    self.data
                        .tree
                        .range(expressions)
                        .map(|i| {
                            let ni = self.data.node_index(i);
                            self.compile_expr(ni).unwrap()
                        })
                        .collect()
                };
                if let Some(merge_block) = self.state.block_expr_stack.last() {
                    let yields = self.state.block_yields.last_mut().unwrap();
                    yields.push((values[0], self.b.get_insert_block().unwrap()));
                    self.b.build_unconditional_branch(*merge_block)?;
                }
            }
            Tag::While => {
                let parent_fn = self.state.function.unwrap();
                let condition_expr = self.compile_expr(node.lhs)?.into_int_value();
                let while_block = self.context.append_basic_block(parent_fn, "while_block");
                let merge_block = self.context.append_basic_block(parent_fn, "merge_block");
                // check condition
                // true? jump to loop body
                // false? jump to after loop
                self.b
                    .build_conditional_branch(condition_expr, while_block, merge_block)?;
                // block_while:
                self.b.position_at_end(while_block);
                let body = self.data.node(node.rhs);
                for i in body.lhs..body.rhs {
                    let ni = self.data.node_index(i);
                    self.compile_stmt(ni)?;
                }
                let condition_expr = self.compile_expr(node.lhs)?.into_int_value();
                // brnz block_while
                self.b
                    .build_conditional_branch(condition_expr, while_block, merge_block)?;
                // block_merge:
                self.b.position_at_end(merge_block);
            }
            _ => {
                self.compile_expr(node_id)?;
            }
        }
        Ok(())
    }

    pub fn compile_exprs(
        &mut self,
        node_id: NodeId,
    ) -> Result<Vec<BasicValueEnum<'ctx>>, BuilderError> {
        let node = self.data.tree.node(node_id);
        assert_eq!(node.tag, Tag::Expressions);
        let range = self.data.tree.range(node);
        let mut values = Vec::with_capacity(range.len());
        for i in range {
            let ni = self.data.tree.node_index(i);
            let value = self.compile_expr(ni)?;
            let type_ids = &[self.data.type_id(ni)];
            let type_ids = if let Typ::Tuple { fields } = self.data.typ(ni) {
                fields.as_slice()
            } else {
                type_ids
            };
            if type_ids.len() == 1 {
                values.push(value);
            } else {
                for i in 0..type_ids.len() {
                    let value = self.b.build_extract_value::<StructValue>(
                        value.into_struct_value(),
                        i as u32,
                        "extract_value",
                    )?;
                    values.push(value);
                }
            }
        }
        Ok(values)
    }

    pub fn compile_expr(&mut self, node_id: NodeId) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let node = *self.data.node(node_id);

        let value_type = llvm_type(self.context, &self.data, self.data.type_id(node_id));

        macro_rules! compile_binary_expr {
            ($inst:ident, $name:literal) => {{
                let (lhs, rhs) = self.compile_children(&node)?;
                Ok(self.b.$inst(lhs, rhs, $name).unwrap().into())
            }};
        }
        macro_rules! compile_int_compare {
            ($predicate:expr, $name:literal) => {{
                let (lhs, rhs) = self.compile_children(&node)?;
                Ok(self
                    .b
                    .build_int_compare($predicate, lhs, rhs, $name)
                    .unwrap()
                    .into())
            }};
        }

        match node.tag {
            // Variables
            Tag::Access => {
                let container = self.compile_expr(node.lhs)?;
                let field_id = self
                    .data
                    .definitions
                    .get_definition_id(node_id, Diagnostic::error())
                    .expect("failed to lookup field definition");
                let field = self.data.node(field_id);
                let field_index = self.data.tree.node_extra(field, 1);
                match container {
                    BasicValueEnum::PointerValue(pointer) => {
                        // Automatically deref.
                        let struct_type = llvm_type(
                            self.context,
                            &self.data,
                            self.data.typ(node.lhs).element_type(),
                        );
                        // Returns the pointer to the struct field.
                        let gep = self
                            .b
                            .build_struct_gep(
                                struct_type.as_basic_type_enum(),
                                pointer,
                                field_index,
                                "",
                            )
                            .unwrap();
                        self.b.build_load(value_type, gep, "")
                    }
                    BasicValueEnum::StructValue(value) => {
                        self.b.build_extract_value(value, field_index, "")
                    }
                    _ => unreachable!("cannot gep_struct for non-struct value"),
                }
            }
            Tag::Address => Ok(self.locate(node.lhs).base.into()),
            Tag::Dereference => {
                let location = self.compile_lvalue(node.lhs)?;
                let ptr_type = llvm_type(self.context, &self.data, self.data.type_id(node.lhs));
                let ptr = self
                    .b
                    .build_load(ptr_type, location, "pointer_value")?
                    .into_pointer_value();
                self.b.build_load(value_type, ptr, "deref")
            }
            Tag::Identifier => {
                let name = self.data.tree.name(node_id);
                let location = self.locate(node_id);
                self.b.build_load(value_type, location.base, name)
            }
            Tag::Subscript => {
                let lvalue = self.compile_lvalue(node_id)?;
                self.b.build_load(value_type, lvalue, "subscript")
            }
            Tag::Conversion => {
                let x = self.compile_expr(node.lhs).unwrap().into_int_value();
                let from_type = llvm_type(self.context, &self.data, self.data.type_id(node.lhs))
                    .into_int_type();
                let to_type =
                    llvm_type(self.context, &self.data, self.data.type_id(node_id)).into_int_type();
                let value = if to_type.get_bit_width() < from_type.get_bit_width() {
                    self.b.build_int_truncate(x, to_type, "trunc")
                } else if self.data.typ(node_id).is_signed() {
                    self.b.build_int_s_extend(x, to_type, "sext")
                } else {
                    self.b.build_int_z_extend(x, to_type, "zext")
                }?;
                Ok(value.as_basic_value_enum())
            }
            Tag::IfxElse => {
                let parent_fn = self.state.function.unwrap();
                let mut if_nodes = Vec::new();
                let mut then_blocks = Vec::new();
                for i in node.lhs..node.rhs {
                    let index = self.data.node_index(i);
                    let if_node = *self.data.node(index);
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
                self.state.block_expr_stack.push(merge_block);
                self.state.block_yields.push(vec![]);
                // Compile branches.
                for i in 0..if_count {
                    let condition_expr = self.compile_expr(if_nodes[i].lhs)?.into_int_value();
                    if i < if_count - 1 {
                        // This is not the last else-if block.
                        let block = self.context.append_basic_block(parent_fn, "block");
                        self.b
                            .build_conditional_branch(condition_expr, then_blocks[i], block)?;
                        self.b.position_at_end(block);
                    } else if !has_else {
                        // This is the last else-if block and there's no else.
                        self.b.build_conditional_branch(
                            condition_expr,
                            then_blocks[i],
                            merge_block,
                        )?;
                    } else {
                        // This is the last else-if block and there's an else.
                        self.b.build_conditional_branch(
                            condition_expr,
                            then_blocks[i],
                            then_blocks[if_count],
                        )?;
                    }
                }
                // Compile block statements.
                for (i, if_node) in if_nodes.iter().enumerate() {
                    self.b.position_at_end(then_blocks[i]);
                    let body = self.data.node(if_node.rhs);
                    for j in body.lhs..body.rhs {
                        let index = self.data.node_index(j);
                        self.compile_stmt(index)?;
                    }
                    if self
                        .b
                        .get_insert_block()
                        .unwrap()
                        .get_terminator()
                        .is_none()
                    {
                        self.b.build_unconditional_branch(merge_block)?;
                    }
                }
                self.b.position_at_end(merge_block);
                let phi = self.b.build_phi(value_type, "phi")?;
                let mut block_yields = Vec::<(&dyn BasicValue, BasicBlock)>::new();
                for i in self.state.block_yields.last().unwrap() {
                    let (value, block) = i;
                    block_yields.push((value, *block));
                }
                // phi.add_incoming(&[(&lhs, lhs_block), (&rhs.into_int_value(), rhs_block)]);
                phi.add_incoming(&block_yields);
                self.state.block_expr_stack.pop();
                // println!("{}", self.module.print_to_string().to_string_lossy());
                self.state.block_yields.pop();
                Ok(phi.as_basic_value())
            }
            // Function calls
            Tag::Add | Tag::Mul => self.compile_call(node_id, node_id, true),
            Tag::Call => self.compile_call(node_id, node.lhs, false),
            // Arithmetic operators
            Tag::Div => compile_binary_expr!(build_int_signed_div, "int_signed_div"),
            Tag::Sub => compile_binary_expr!(build_int_sub, "int_sub"),
            Tag::Negation => {
                if self.data.tree.node(node.lhs).tag == Tag::IntegerLiteral {
                    return Ok(self.compile_integer_literal(
                        node.lhs,
                        self.data.type_id(node_id),
                        true,
                    ));
                }
                let value = self.compile_expr(node.lhs)?.into_int_value();
                Ok(self.b.build_int_neg(value, "int_neg")?.into())
            }
            // Bitwise operators
            Tag::BitwiseShiftL => compile_binary_expr!(build_left_shift, "left_shift"),
            Tag::BitwiseShiftR => {
                let (lhs, rhs) = self.compile_children(&node)?;
                Ok(self
                    .b
                    .build_right_shift(lhs, rhs, true, "right_shift")?
                    .into())
            }
            Tag::BitwiseXor => compile_binary_expr!(build_xor, "xor"),
            // Logical operators
            Tag::Equality => compile_int_compare!(IntPredicate::EQ, "int_compare_eq"),
            Tag::Inequality => compile_int_compare!(IntPredicate::NE, "int_compare_ne"),
            Tag::Greater => compile_int_compare!(IntPredicate::SGT, "int_compare_sgt"),
            Tag::GreaterEqual => compile_int_compare!(IntPredicate::SGE, "int_compare_sge"),
            Tag::Less => compile_int_compare!(IntPredicate::SLT, "int_compare_slt"),
            Tag::LessEqual => compile_int_compare!(IntPredicate::SLE, "int_compare_sle"),
            Tag::LogicalAnd => self.compile_short_circuit(&node, true),
            Tag::LogicalOr => self.compile_short_circuit(&node, false),
            Tag::Not => {
                let value = self.compile_expr(node.lhs)?.into_int_value();
                Ok(self.b.build_not(value, "not")?.into())
            }
            // Literal values
            Tag::False => Ok(self.context.bool_type().const_int(0, false).into()),
            Tag::True => Ok(self.context.bool_type().const_int(1, false).into()),
            Tag::FloatLiteral => {
                let token_str = self.data.tree.node_lexeme(node_id);
                // dbg!(token_str);
                let value = token_str.parse::<f32>().unwrap();
                // dbg!(value);
                Ok(self
                    .context
                    .f32_type()
                    .const_float(value as f64)
                    .as_basic_value_enum())
            }
            Tag::IntegerLiteral => {
                Ok(self.compile_integer_literal(node_id, self.data.type_id(node_id), false))
            }
            Tag::StringLiteral => {
                let mut string = self
                    .data
                    .tree
                    .token_str(node.token)
                    .trim_matches('"')
                    .to_string();
                let length = string.len();
                string.push('\0');
                let ptr = self
                    .b
                    .build_global_string_ptr(&string, "string_data")
                    .unwrap();
                let len = self.context.i64_type().const_int(length as u64, true);
                Ok(self
                    .context
                    .const_struct(
                        &[ptr.as_basic_value_enum(), len.as_basic_value_enum()],
                        false,
                    )
                    .into())
            }
            _ => unreachable!("Invalid expression tag: {:?}", node.tag),
        }
    }

    fn compile_lvalue(&mut self, node_id: NodeId) -> Result<PointerValue<'ctx>, BuilderError> {
        let node = *self.data.node(node_id);
        match node.tag {
            // a.x = b
            Tag::Access => {
                // Return the location of the struct field.
                let field_id = self
                    .data
                    .definitions
                    .get_definition_id(node_id, Diagnostic::error())
                    .expect("failed to lookup field definition");
                let field = self.data.node(field_id);
                let field_index = self.data.tree.node_extra(field, 1);
                let mut struct_ptr = self.compile_lvalue(node.lhs)?;
                let a_type = llvm_type(self.context, &self.data, self.data.type_id(node.lhs));
                let struct_ty = if let Typ::Pointer { typ, .. } = self.data.typ(node.lhs) {
                    // Dereference pointer values.
                    let load = self.b.build_load(a_type, struct_ptr, "deref")?;
                    struct_ptr = load.into_pointer_value();
                    llvm_type(self.context, &self.data, *typ)
                } else {
                    a_type
                };
                self.b
                    .build_struct_gep(struct_ty, struct_ptr, field_index, "")
            }
            // @a = b
            Tag::Dereference => {
                // Return the pointer location.
                let location = self.compile_lvalue(node.lhs)?;
                let a_type = llvm_type(self.context, &self.data, self.data.type_id(node.lhs));
                Ok(self
                    .b
                    .build_load(a_type, location, "deref_lvalue")?
                    .into_pointer_value())
            }
            // a = b
            Tag::Identifier => Ok(self.locate(node_id).base),
            // a[i] = b
            Tag::Subscript => {
                // Return the location of the array element.
                let array_ptr = self.compile_lvalue(node.lhs)?;
                let index_value = self.compile_expr(node.rhs)?.into_int_value();
                // Offset for 1-based indexing.
                let base_offset = self.context.i64_type().const_int(1, false);
                let index_value = self
                    .b
                    .build_int_sub(index_value, base_offset, "int_sub")
                    .unwrap();
                let element_type = llvm_type(self.context, &self.data, self.data.type_id(node_id));
                unsafe {
                    self.b
                        .build_gep(element_type, array_ptr, &[index_value], "gep")
                }
            }
            _ => unreachable!("Invalid lvalue {:?} for assignment", node.tag),
        }
    }

    fn compile_children(
        &mut self,
        node: &Node,
    ) -> Result<(IntValue<'ctx>, IntValue<'ctx>), BuilderError> {
        let lhs = self.compile_expr(node.lhs)?.into_int_value();
        let rhs = self.compile_expr(node.rhs)?.into_int_value();
        Ok((lhs, rhs))
    }

    ///
    /// 1. Allocate space for returns.
    /// 2. Copy arguments.
    /// 3. Call function.
    /// 3.
    fn compile_call(
        &mut self,
        node_id: NodeId,
        callee_id: NodeId,
        binary: bool,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let node = self.data.tree.node(node_id);
        let mut type_arguments = vec![];
        if !binary {
            let callee = self.data.node(node.lhs);
            for i in self.data.tree.range(callee) {
                let ni = self.data.tree.node_index(i);
                type_arguments.push(self.data.type_id(ni));
            }
        }
        let arg_ids = if binary {
            vec![node.lhs, node.rhs]
        } else {
            self.data
                .tree
                .range(self.data.node(node.rhs))
                .map(|i| self.data.node_index(i))
                .collect()
        };

        let definition = self.data.definitions.get(&callee_id).unwrap_or_else(|| {
            panic!("Definition not found: {}", "failed to get function decl id")
        });

        let name = match &definition {
            Definition::BuiltInFunction(id) => {
                return self.compile_built_in_function(*id, node_id);
            }
            Definition::User(id) | Definition::Foreign(id) | Definition::Overload(id) => {
                let arg_type_ids: Vec<TypeId> =
                    arg_ids.iter().map(|ni| self.data.type_id(*ni)).collect();
                if let Some(map) = self.data.type_parameters.get(id) {
                    let key = if !type_arguments.is_empty() {
                        &type_arguments
                    } else {
                        &arg_type_ids
                    };
                    let solution = map.get(key).unwrap();
                    self.data
                        .mangle_function_declaration(*id, true, solution, None)
                } else {
                    self.data
                        .mangle_function_declaration(*id, true, &[], Some(&arg_type_ids))
                }
            }
            Definition::Resolved(id) => self.data.mangle_function_declaration(*id, true, &[], None),
            Definition::BuiltInType(built_in_type) => {
                let node = self.data.tree.node(node_id);
                let args = self.data.node(node.rhs);
                let ni = self.data.node_index(args.lhs);
                let arg = self.compile_expr(ni)?;

                let from_type_enum = llvm_type(self.context, &self.data, self.data.type_id(ni));
                let from_bits = from_type_enum.into_int_type().get_bit_width();

                let ti = *built_in_type as TypeId;
                let to_typ = &self.data.types[ti];
                let to_type_enum = llvm_type(self.context, &self.data, ti);
                let to_bits = to_type_enum.into_int_type().get_bit_width();

                return if arg.get_type().is_float_type() {
                    let value = arg.into_float_value();
                    let to_type = to_type_enum.into_float_type();
                    Ok(if to_bits < from_bits {
                        self.b.build_float_trunc(value, to_type, "trunc")
                    } else {
                        self.b.build_float_ext(value, to_type, "ext")
                    }?
                    .as_basic_value_enum())
                } else {
                    let value = arg.into_int_value();
                    let to_type = to_type_enum.into_int_type();
                    Ok(if to_bits < from_bits {
                        self.b.build_int_truncate(value, to_type, "trunc")
                    } else if to_typ.is_signed() {
                        self.b.build_int_s_extend(value, to_type, "sext")
                    } else {
                        self.b.build_int_z_extend(value, to_type, "sext")
                    }?
                    .as_basic_value_enum())
                };
            }
            _ => unreachable!("Definition not found: {}", "failed to get function decl id"),
        };

        let callee = self
            .module
            .get_function(&name)
            .unwrap_or_else(|| panic!("failed to get module function \"{}\"", &name));

        // Assume one return value.
        let return_type = self.data.type_id(node_id);

        let results: Result<Vec<BasicValueEnum>, BuilderError> = arg_ids
            .iter()
            .map(|ni| -> Result<BasicValueEnum<'ctx>, BuilderError> { self.compile_expr(*ni) })
            .collect();
        let args = results?;

        let mut argiter = args.iter();
        let argslice = argiter.by_ref();
        let argslice = argslice
            .map(|&val| val.into())
            .collect::<Vec<BasicMetadataValueEnum>>();
        let argslice = argslice.as_slice();
        let call_site_value = self.b.build_call(callee, argslice, &name)?;
        if return_type != T::Void as TypeId {
            Ok(call_site_value
                .try_as_basic_value()
                .left()
                .expect("basic value expected"))
        } else {
            Ok(self.context.const_struct(&[], false).as_basic_value_enum())
        }
    }

    fn compile_built_in_function(
        &mut self,
        built_in_function: BuiltInFunction,
        node_id: NodeId,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let node = self.data.tree.node(node_id);
        match built_in_function {
            BuiltInFunction::Add | BuiltInFunction::AddI8 | BuiltInFunction::AddI32 => {
                let a = self.compile_expr(node.lhs)?.into_int_value();
                let b = self.compile_expr(node.rhs)?.into_int_value();
                Ok(self.b.build_int_add(a, b, "int_add")?.into())
            }
            BuiltInFunction::Mul => {
                let a = self.compile_expr(node.lhs)?.into_int_value();
                let b = self.compile_expr(node.rhs)?.into_int_value();
                Ok(self.b.build_int_mul(a, b, "int_mul")?.into())
            }
            BuiltInFunction::SizeOf => {
                let args = self.data.tree.node(node.rhs);
                let first_arg_id = self.data.tree.node_index(args.lhs);
                let type_id = self.data.type_id(first_arg_id);
                let value = sizeof(self.data.types, type_id) as u64;
                Ok(self.context.i64_type().const_int(value, false).into())
            }
            BuiltInFunction::SubI64 => {
                let a = self.compile_expr(node.lhs)?.into_int_value();
                let b = self.compile_expr(node.rhs)?.into_int_value();
                Ok(self.b.build_int_sub(a, b, "int_sub")?.into())
            }
        }
    }

    fn compile_short_circuit(
        &mut self,
        node: &Node,
        is_and: bool,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let parent_fn = self.state.function.unwrap();
        let lhs = self.compile_expr(node.lhs)?.into_int_value();
        // Record the first predecessor block after compiling the lhs.
        let lhs_block = self.b.get_insert_block().unwrap();
        let right_block = self.context.append_basic_block(parent_fn, "right_block");
        let merge_block = self.context.append_basic_block(parent_fn, "merge_block");
        if is_and {
            self.b
                .build_conditional_branch(lhs, right_block, merge_block)?;
        } else {
            self.b
                .build_conditional_branch(lhs, merge_block, right_block)?;
        }
        self.b.position_at_end(right_block);
        let rhs = self.compile_expr(node.rhs)?;
        // Record the second predecessor block after compiling the lhs.
        let rhs_block = self.b.get_insert_block().unwrap();
        self.b.build_unconditional_branch(merge_block)?;
        self.b.position_at_end(merge_block);
        let phi = self.b.build_phi(self.context.bool_type(), "phi")?;
        phi.add_incoming(&[(&lhs, lhs_block), (&rhs.into_int_value(), rhs_block)]);
        Ok(phi.as_basic_value())
    }

    fn compile_integer_literal(
        &self,
        node_id: NodeId,
        type_id: TypeId,
        negative: bool,
    ) -> BasicValueEnum<'ctx> {
        let token_str = self.data.tree.node_lexeme(node_id);
        let value = token_str.parse::<i64>().unwrap();
        let llvm_type = llvm_type(self.context, &self.data, type_id);
        llvm_type
            .into_int_type()
            .const_int(if negative { -value } else { value } as u64, true)
            .into()
    }

    fn locate(&self, node_id: NodeId) -> Location<'ctx> {
        let node = self.data.node(node_id);
        match node.tag {
            Tag::Identifier => {
                let def_id = self
                    .data
                    .definitions
                    .get_definition_id(node_id, Diagnostic::error())
                    .expect("failed to lookup variable definition");
                *self
                    .state
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

pub fn llvm_type<'ctx>(context: &'ctx Context, data: &Data, type_id: usize) -> BasicTypeEnum<'ctx> {
    let types = &data.types;
    match &types[type_id] {
        Typ::Array { typ, length, .. } => llvm_type(context, data, *typ)
            .array_type(*length as u32)
            .into(),
        Typ::Pointer { typ, .. } => llvm_type(context, data, *typ)
            .ptr_type(AddressSpace::default())
            .into(),
        Typ::Struct { fields, .. } => {
            let mut field_types = Vec::with_capacity(fields.len());
            for id in fields {
                field_types.push(llvm_type(context, data, *id))
            }
            context.struct_type(&field_types, false).into()
        }
        Typ::Void | Typ::None => context.struct_type(&[], false).into(),
        Typ::Boolean => context.bool_type().into(),
        Typ::Numeric {
            floating, bytes, ..
        } => match (*floating, *bytes) {
            (true, 4) => context.f32_type().into(),
            (true, 8) => context.f64_type().into(),
            (false, 1) => context.i8_type().into(),
            (false, 2) => context.i16_type().into(),
            (false, 4) => context.i32_type().into(),
            (false, 8) => context.i64_type().into(),
            _ => unreachable!(
                "Invalid numeric type: {:?} is not a valid LLVM type.",
                &types[type_id]
            ),
        },
        Typ::Tuple { fields, .. } => {
            let mut field_types = Vec::with_capacity(fields.len());
            for id in fields {
                field_types.push(llvm_type(context, data, *id))
            }
            context.struct_type(&field_types, false).into()
        }
        Typ::TypeParameter { index, .. } => llvm_type(
            context,
            data,
            data.active_type_parameters.last().unwrap()[*index],
        ),
        Typ::Parameter { binding, .. } => llvm_type(context, data, *binding),
        _ => unreachable!(
            "Invalid type: {:?} is not a valid LLVM type.",
            &types[type_id]
        ),
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Location<'ctx> {
    base: PointerValue<'ctx>,
    _offset: i32,
}

impl<'ctx> Location<'ctx> {
    fn new(base: PointerValue<'ctx>, _offset: i32) -> Self {
        Self { base, _offset }
    }
    // fn offset(self, extra_offset: i32) -> Self {
    //     Location {
    //         base: self.base,
    //         offset: self.offset + extra_offset,
    //     }
    // }
}
