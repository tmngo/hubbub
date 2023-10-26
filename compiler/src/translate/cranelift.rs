use crate::{
    analyze::{BuiltInFunction, Definition, Lookup},
    builtin,
    parse::{Node, NodeId, NodeInfo, Tag},
    translate::input::{sizeof, Data, Input, Layout, Shape},
    types::{Type as Typ, TypeId},
    workspace::Workspace,
};
use codespan_reporting::diagnostic::Diagnostic;
use cranelift::{
    codegen::ir::ArgumentPurpose,
    prelude::{
        codegen::{ir::StackSlot, Context},
        isa::{lookup, TargetFrontendConfig},
        settings,
        types::{F32, F64, I16, I32, I64, I8},
        AbiParam, Block, Configurable, FunctionBuilder, FunctionBuilderContext, InstBuilder, IntCC,
        MemFlags, Signature, StackSlotData, StackSlotKind, Type, Value, Variable,
    },
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

// ModuleCompiler: FunctionBuilderContext, Context, pointer_type
// - create builder from context, function builder context
// - compile function
// - module.define_function()
// - module.clear_context()
pub struct ModuleCompiler<'a> {
    pub builder_ctx: FunctionBuilderContext,
    pub ctx: Context,
    data_ctx: DataContext,
    data: Data<'a>,
    pub state: State,
}

impl<'a> ModuleCompiler<'a> {
    pub fn new(
        workspace: &Workspace,
        input: &'a Input<'a>,
        output_name: String,
        use_jit: bool,
    ) -> Self {
        let mut flag_builder = settings::builder();
        // flag_builder.set("enable_verifier", "false").ok();
        flag_builder.set("is_pic", "true").unwrap();
        let isa_builder = lookup(target_lexicon::HOST).unwrap();
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();

        let module: Box<dyn CraneliftModule> = if use_jit {
            let mut jit_builder =
                JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
            jit_builder.symbols(vec![
                ("Base.print_int", builtin::print_int as *const u8),
                ("Base.print_f32", builtin::print_f32 as *const u8),
                ("Base.print_f64", builtin::print_f64 as *const u8),
                ("Base.print_cstr", builtin::print_cstr as *const u8),
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
            .map(|(t, _)| Layout::new(input.types, t))
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
                block_expr_stack: vec![],
                signatures: HashMap::new(),
                // func_refs: HashMap::new(),
            },
        }
    }

    ///
    pub fn compile_nodes(mut self, filename: &Path) -> Option<i64> {
        let tree = self.data.tree;
        let root = tree.node(0);
        let mut fn_ids = Vec::new();
        let mut main_id = None;

        // Compile function signatures.
        for i in root.lhs..root.rhs {
            let module_index = self.data.node_index(i);
            let module = self.data.node(module_index);
            if module.tag != Tag::Module {
                continue;
            }

            for i in module.lhs..module.rhs {
                let ni = tree.node_index(i);
                let node = tree.node(ni);
                if let Tag::FunctionDecl = node.tag {
                    if self.data.node(node.lhs).lhs != 0 {
                        // Skip generic functions with no specializations.
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
                            self.compile_function_signature(ni, &name);
                            self.data.active_type_parameters.pop();
                        }
                        continue;
                    }
                    if node.rhs == 0 {
                        // continue;
                    }
                    let name = self.data.mangle_function_declaration(ni, true, &[], None);
                    self.compile_function_signature(ni, &name);
                };
            }
        }

        // Compile function bodies.
        for i in root.lhs..root.rhs {
            let module_index = self.data.node_index(i);
            let module = *self.data.node(module_index);
            if module.tag != Tag::Module {
                continue;
            }

            for i in module.lhs..module.rhs {
                let ni = self.data.node_index(i);
                let node = tree.node(ni);
                if let Tag::FunctionDecl = node.tag {
                    // Skip generic functions with no specializations.
                    // dbg!(self.data.tree.name(ni));

                    if self.data.node(node.lhs).lhs != 0 {
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
                            self.compile_function_decl(ni, name);
                            self.data.active_type_parameters.pop();
                        }
                        continue;
                    }
                    // Skip function signatures
                    if node.rhs == 0 {
                        continue;
                    }
                    let name = self.data.mangle_function_declaration(ni, true, &[], None);
                    let fn_id = self.compile_function_decl(ni, name);
                    let fn_name = self.data.tree.name(ni);
                    if fn_name == "main" {
                        fn_ids.push(fn_id);
                        main_id = Some(fn_id);
                    }
                };
            }
        }
        if let Some(id) = main_id {
            let code = self.state.module.finalize(id, filename);
            return Some(code);
        }
        None
    }

    ///
    pub fn compile_function_decl(&mut self, node_id: NodeId, name: String) -> FuncId {
        let target_config = self.state.module.target_config();
        let ptr_type = target_config.pointer_type();

        let mut c = FunctionCompiler {
            b: FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx),
            data_ctx: &mut self.data_ctx,
            ptr_type,
            target_config,
            data: &self.data,
            state: &mut self.state,
        };

        // Build signature
        // let node = data.node(node_id);
        // let fn_id = Self::compile_function_signature(state, data, self.b.func, node.lhs, &name);

        // Get cached signature
        let cached_func = c.state.signatures.get(&name).unwrap().clone();
        let fn_id = cached_func.0;
        c.b.func.signature = cached_func.1;

        c.compile_function_body(node_id);
        c.b.finalize();
        // println!("{} :: {}", name, c.b.func.display());

        // Write CLIF to file
        // let mut s = String::new();
        // cranelift::codegen::write_function(&mut s, self.b.func);
        // std::fs::write(
        //     format!("./fn-{}.clif", name.replace("|", "-").replace(",", "-"),),
        //     s,
        // );

        self.state
            .module
            .define_function(fn_id, &mut self.ctx)
            .unwrap();
        // println!("{}", cranelift::codegen::timing::take_current());
        self.state.module.clear_context(&mut self.ctx);
        self.state.filled_blocks.clear();
        fn_id
    }

    fn compile_function_signature(&mut self, node_id: NodeId, name: &str) -> FuncId {
        let fn_decl = self.data.node(node_id);
        let signature_type = self.data.typ(fn_decl.lhs);
        let signature =
            self.create_cranelift_signature(signature_type.parameters(), signature_type.returns());
        let linkage = if let Some(NodeInfo::Prototype { foreign, .. }) =
            self.data.tree.info.get(&fn_decl.lhs)
        {
            if *foreign || fn_decl.rhs == 0 {
                Linkage::Import
            } else {
                Linkage::Export
            }
        } else {
            Linkage::Export
        };
        let fn_id = self
            .state
            .module
            .declare_function(name, linkage, &signature)
            .unwrap();
        self.state
            .signatures
            .insert(name.to_string(), (fn_id, signature));
        fn_id
    }

    fn create_cranelift_signature(
        &self,
        param_types: &[TypeId],
        return_types: &[TypeId],
    ) -> Signature {
        let ptr_type = self.state.module.target_config().pointer_type();
        let mut signature = self.state.module.make_signature();
        for ti in param_types.iter() {
            signature.params.push(AbiParam::new(cl_type(
                &self.data,
                ptr_type,
                &self.data.types[*ti],
            )))
        }
        let mut return_bytes = 0;
        for ti in return_types.iter() {
            let size = sizeof(self.data.types, *ti);
            return_bytes += size;
        }
        if return_bytes > 8 || return_types.len() > 1 {
            signature
                .params
                .push(AbiParam::special(ptr_type, ArgumentPurpose::StructReturn))
        } else {
            for ti in return_types.iter() {
                signature.returns.push(AbiParam::new(cl_type(
                    &self.data,
                    ptr_type,
                    &self.data.types[*ti],
                )))
            }
        }
        signature
    }
}

// FunctionCompiler: FunctionBuilder, pointer_type
// - create blocks
// - finalize
pub struct FunctionCompiler<'a> {
    b: FunctionBuilder<'a>,
    data_ctx: &'a mut DataContext,
    ptr_type: Type,
    target_config: TargetFrontendConfig,
    data: &'a Data<'a>,
    state: &'a mut State,
}

impl FunctionCompiler<'_> {
    fn compile_function_body(&mut self, node_id: NodeId) {
        let fn_decl = *self.data.node(node_id);
        let prototype = self.data.node(fn_decl.lhs);
        let parameters = *self.data.node(self.data.tree.node_extra(prototype, 0));
        let body = *self.data.node(fn_decl.rhs);
        assert_eq!(fn_decl.tag, Tag::FunctionDecl);
        assert_eq!(body.tag, Tag::Block);

        let entry_block = self.b.create_block();
        self.b.func.layout.append_block(entry_block);
        self.b.append_block_params_for_function_params(entry_block);
        self.b.switch_to_block(entry_block);
        self.b.seal_block(entry_block);

        // Define parameters as stack variables.
        for i in parameters.lhs..parameters.rhs {
            let ni = self.data.node_index(i);
            let stack_slot = self.create_stack_slot(ni);
            let location = Location::stack(stack_slot, 0);
            self.state.locations.insert(ni, location);
            let parameter_index = (i - parameters.lhs) as usize;
            let value = self.b.block_params(entry_block)[parameter_index];
            let layout = self.data.layout(ni);
            self.store(&location, value, MemFlags::new(), layout);
        }

        for i in body.lhs..body.rhs {
            let ni = self.data.node_index(i);
            self.compile_stmt(ni);
        }
        // let last_inst = self.b.func.layout.last_inst(self.b.current_block().unwrap()).unwrap();
        // let is_filled = self.b.func.dfg
        if self.data.typ(fn_decl.lhs).returns().is_empty()
            && !self
                .state
                .filled_blocks
                .contains(&self.b.current_block().unwrap())
        {
            self.b.ins().return_(&[]);
            self.state
                .filled_blocks
                .insert(self.b.current_block().unwrap());
        }
    }
    ///
    pub fn compile_stmt(&mut self, node_id: NodeId) {
        let node = *self.data.node(node_id);
        match node.tag {
            Tag::Assign => {
                // lhs: expr
                // rhs: expr
                let layout = self.data.layout(node.rhs);
                let rvalue = self.compile_expr_value(node.rhs);
                let flags = MemFlags::new();
                let lvalue = self.compile_lvalue(node.lhs);
                self.store(&lvalue, rvalue, flags, layout);
            }
            Tag::If => {
                // Each basic block must be filled
                // i.e. end in a terminator: brtable, jump, return, or trap
                // Conditional branch instructions must be followed by a terminator.
                let condition_expr = self.compile_expr_value(node.lhs);
                let then_block = self.b.create_block();
                let merge_block = self.b.create_block();
                self.b.ins().brz(condition_expr, merge_block, &[]);
                self.b.ins().jump(then_block, &[]);
                self.b.seal_block(then_block);
                self.state
                    .filled_blocks
                    .insert(self.b.current_block().unwrap());
                // then block
                self.b.switch_to_block(then_block);
                let body = self.data.node(node.rhs);
                for i in body.lhs..body.rhs {
                    let index = self.data.node_index(i);
                    self.compile_stmt(index);
                }
                // Check if the last statement compiled was a terminator.
                if !self
                    .state
                    .filled_blocks
                    .contains(&self.b.current_block().unwrap())
                {
                    self.b.ins().jump(merge_block, &[]);
                    self.state
                        .filled_blocks
                        .insert(self.b.current_block().unwrap());
                }
                self.b.seal_block(merge_block);
                // merge block
                self.b.switch_to_block(merge_block);
            }
            Tag::IfElse => {
                let mut if_nodes = Vec::new();
                let mut then_blocks = Vec::new();
                for i in node.lhs..node.rhs {
                    let index = self.data.node_index(i);
                    let if_node = self.data.node(index);
                    if_nodes.push(if_node);
                    then_blocks.push(self.b.create_block());
                }
                // If the last else-if block has no condition, it's an else.
                let has_else = if_nodes.last().unwrap().lhs == 0;
                let if_count = if has_else {
                    if_nodes.len() - 1
                } else {
                    if_nodes.len()
                };
                let merge_block = self.b.create_block();
                // Compile branches.
                for i in 0..if_count {
                    let condition_expr = self.compile_expr_value(if_nodes[i].lhs);
                    self.b.ins().brnz(condition_expr, then_blocks[i], &[]);
                    self.b.seal_block(then_blocks[i]);
                    if i < if_count - 1 {
                        // This is not the last else-if block.
                        let block = self.b.create_block();
                        self.b.ins().jump(block, &[]);
                        self.b.seal_block(block);
                        self.b.switch_to_block(block);
                    } else if !has_else {
                        // This is the last else-if block and there's no else.
                        self.b.ins().jump(merge_block, &[]);
                    } else {
                        // This is the last else-if block and there's an else.
                        self.b.ins().jump(then_blocks[if_count], &[]);
                        self.b.seal_block(then_blocks[if_count]);
                    }
                }
                // Compile block statements.
                for (i, if_node) in if_nodes.iter().enumerate() {
                    self.b.switch_to_block(then_blocks[i]);
                    let body = self.data.node(if_node.rhs);
                    for j in body.lhs..body.rhs {
                        let index = self.data.node_index(j);
                        self.compile_stmt(index);
                    }
                    if !self
                        .state
                        .filled_blocks
                        .contains(&self.b.current_block().unwrap())
                    {
                        self.b.ins().jump(merge_block, &[]);
                    }
                }
                self.b.seal_block(merge_block);
                self.b.switch_to_block(merge_block);
            }
            Tag::VariableDecl => {
                // lhs: expressions
                // rhs: expressions
                // There should be one stack slot per thing on the RHS.
                let identifiers = self.data.tree.node(node.lhs);
                let rvalues_id = self.data.tree.node_extra(&node, 1);

                assert_eq!(identifiers.tag, Tag::Expressions);
                let mut locs = vec![];
                let mut identifier_ids = vec![];

                // If no rhs, create stack slots based on the identifier types.
                if rvalues_id == 0 {
                    // let mut ltypes = vec![];
                    for i in self.data.tree.range(identifiers) {
                        let ni = self.data.tree.node_index(i);
                        identifier_ids.push(ni);
                        // ltypes.push(self.data.type_id(ni));
                        let slot = self.create_stack_slot(ni);
                        let location = Location::stack(slot, 0);
                        locs.push(location);

                        // Associate the stack slots with the identifier locations.
                        self.state.locations.insert(ni, location);
                    }
                    return;
                }

                for i in self.data.tree.range(identifiers) {
                    let ni = self.data.tree.node_index(i);
                    identifier_ids.push(ni);
                }

                // Otherwise, create stack slots based on the expression types.

                let expressions = self.data.node(rvalues_id);
                // let mut values = Vec::with_capacity(range.len());
                let mut identifier_index = 0;
                for i in self.data.tree.range(expressions) {
                    let ni = self.data.tree.node_index(i);
                    // let location = locs[location_index];

                    let slot = self.create_stack_slot(ni);
                    let location = Location::stack(slot, 0);
                    // locs.push(location);

                    let types_per_expr = match self.data.typ(ni) {
                        Typ::Tuple { fields } => fields.len(),
                        Typ::Void => 0,
                        _ => 1,
                    };

                    // Get scalar values to store in identifiers
                    let val = self.compile_expr(ni, Some(location));
                    let expr_values = match val {
                        // Function call
                        Val::Multiple(xs) => xs,
                        // Single value
                        Val::Scalar(_) => {
                            let value = self.cranelift_value(val, ni);
                            vec![value]
                        }
                        Val::Reference { .. } => {
                            vec![]
                        }
                    };

                    let mut location_offset = 0;
                    for (index, id) in identifier_ids
                        .iter()
                        .skip(identifier_index)
                        .take(types_per_expr)
                        .enumerate()
                    {
                        let identifier_location = Location::stack(slot, location_offset);
                        self.state.locations.insert(*id, identifier_location);
                        let layout = self.data.layout(*id);
                        // Only store if not an out param that's already been written.
                        if !expr_values.is_empty() {
                            self.store(
                                &identifier_location,
                                expr_values[index],
                                MemFlags::new(),
                                layout,
                            );
                        }
                        location_offset += self.data.sizeof(self.data.type_id(*id)) as i32;
                    }
                    identifier_index += types_per_expr;
                }

                let rhs = self.data.tree.node(rvalues_id);
                assert_eq!(rhs.tag, Tag::Expressions);
                // let mut location_index = 0;
                // if let Val::Multiple(values) = self.compile_expr(rvalues_id, None) {
                // let type_ids = &[self.data.type_id(rvalues_id)];
                // let type_ids = if let Typ::Tuple { fields } = self.data.typ(rvalues_id) {
                //     fields.as_slice()
                // } else {
                //     type_ids
                // };
                // assert_eq!(type_ids.len(), values.len());
                // for (i, value) in values.iter().enumerate() {
                //     let layout = &self.data.layouts[type_ids[i]];
                //     self.store(&locs[i], *value, MemFlags::new(), layout);
                // }
                // } else {
                //     unreachable!("rhs of variable declaration must be multiple-valued")
                // }
            }
            Tag::Return => {
                let mut return_values = vec![];
                let mut return_types = vec![];

                if node.lhs != 0 {
                    let expressions = self.data.tree.node(node.lhs);
                    for i in self.data.tree.range(expressions) {
                        let ni = self.data.node_index(i);
                        // let value = self.compile_expr_value(ni);
                        let ti = self.data.type_id(ni);
                        // return_values.push(value);
                        return_types.push(ti);
                    }
                };
                if node.lhs != 0 {
                    let (is_struct_return, _) = is_struct_return(self.data, &return_types);
                    if is_struct_return {
                        let entry_block = self.b.func.layout.entry_block().unwrap();
                        let dest = *self.b.block_params(entry_block).last().unwrap();

                        // let stack_slot = self.b.create_sized_stack_slot(StackSlotData {
                        //     kind: StackSlotKind::ExplicitSlot,
                        //     size: return_bytes,
                        // });
                        // let stack_slot_addr = self.b.ins().stack_addr(self.ptr_type, stack_slot, 0);
                        let expressions = self.data.tree.node(node.lhs);
                        let mut offset = 0;
                        for i in self.data.tree.range(expressions) {
                            let ni = self.data.node_index(i);
                            let layout = self.data.layout(ni);
                            let location = Location::pointer(dest, offset);
                            let value = self.compile_expr_value(ni);
                            self.store(&location, value, MemFlags::new(), layout);
                            offset += layout.size as i32;
                        }

                        // let aggregate_size = self.b.ins().iconst(self.ptr_type, return_bytes as i64);

                        // self.b.call_memcpy(
                        //     self.state.module.target_config(),
                        //     dest,
                        //     stack_slot_addr,
                        //     aggregate_size,
                        // );

                        // self.b.ins().return_(&[]);
                    } else {
                        let expressions = self.data.tree.node(node.lhs);

                        // let mut return_values = vec![];
                        for i in self.data.tree.range(expressions) {
                            let ni = self.data.node_index(i);
                            let value = self.compile_expr_value(ni);
                            return_values.push(value);
                        }
                    };
                }

                self.b.ins().return_(&return_values);
                self.state
                    .filled_blocks
                    .insert(self.b.current_block().unwrap());
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
                            self.compile_expr_value(ni)
                        })
                        .collect()
                };
                if let Some(block) = self.state.block_expr_stack.last() {
                    self.b.ins().jump(*block, &values);
                    self.state
                        .filled_blocks
                        .insert(self.b.current_block().unwrap());
                }
            }
            Tag::While => {
                let condition = self.compile_expr_value(node.lhs);
                let while_block = self.b.create_block();
                let merge_block = self.b.create_block();
                // check condition
                // true? jump to loop body
                self.b.ins().brnz(condition, while_block, &[]);
                // false? jump to after loop
                self.b.ins().jump(merge_block, &[]);
                self.state
                    .filled_blocks
                    .insert(self.b.current_block().unwrap());
                // block_while:
                self.b.switch_to_block(while_block);
                let body = self.data.node(node.rhs);
                for i in body.lhs..body.rhs {
                    let ni = self.data.node_index(i);
                    self.compile_stmt(ni);
                }
                let condition = self.compile_expr_value(node.lhs);
                // brnz block_while
                self.b.ins().brnz(condition, while_block, &[]);
                self.b.seal_block(while_block);
                self.b.ins().jump(merge_block, &[]);
                self.b.seal_block(merge_block);
                self.state
                    .filled_blocks
                    .insert(self.b.current_block().unwrap());
                // block_merge:
                self.b.switch_to_block(merge_block);
            }
            Tag::Break => {}
            _ => {
                self.compile_expr(node_id, None);
            }
        }
    }

    pub fn compile_expr_value(&mut self, node_id: NodeId) -> Value {
        let val = self.compile_expr(node_id, None);
        self.cranelift_value(val, node_id)
    }

    /// Returns a value. A value can be a scalar or an aggregate.
    /// NodeId -> Val
    pub fn compile_expr(&mut self, node_id: NodeId, dest: Option<Location>) -> Val {
        let ty = self.ptr_type;
        let node = self.data.node(node_id);

        macro_rules! compile_binary_expr {
            ($inst:ident) => {{
                let (lhs, rhs) = self.compile_children(node);
                Val::Scalar(self.b.ins().$inst(lhs, rhs))
            }};
        }
        macro_rules! compile_int_compare {
            ($cond:expr) => {{
                let (lhs, rhs) = self.compile_children(node);
                let value = self.b.ins().icmp($cond, lhs, rhs);
                Val::Scalar(self.b.ins().bint(I8, value))
            }};
        }

        match node.tag {
            // Variables
            Tag::Access => {
                let loc = self.locate_field(node_id);
                self.load_val(loc, node_id)
            }
            Tag::Address => {
                let loc = self.locate(node.lhs);
                Val::Scalar(self.get_addr(&loc))
            }
            Tag::Dereference => {
                let flags = MemFlags::new();
                let ptr_layout = &self.data.layout2(self.data.type_id(node_id));
                // Load pointer
                let ptr = self.locate(node.lhs);
                let ptr = self.load_value(&ptr, flags, self.data.typ(node.lhs), ptr_layout);
                // Load value
                self.load_val(Location::pointer(ptr, 0), node_id)
            }
            Tag::Expressions => {
                let range = self.data.tree.range(node);
                let mut values = Vec::with_capacity(range.len());
                for i in range {
                    let ni = self.data.tree.node_index(i);
                    let val = self.compile_expr(ni, None);
                    match val {
                        Val::Multiple(xs) => values.extend(xs),
                        _ => {
                            let value = self.cranelift_value(val, ni);
                            values.push(value);
                        }
                    }
                }
                Val::Multiple(values)
            }
            Tag::IfxElse => {
                let mut if_nodes = Vec::new();
                let mut then_blocks = Vec::new();
                for i in node.lhs..node.rhs {
                    let index = self.data.node_index(i);
                    let if_node = self.data.node(index);
                    if_nodes.push(if_node);
                    then_blocks.push(self.b.create_block());
                }
                // If the last else-if block has no condition, it's an else.
                let has_else = if_nodes.last().unwrap().lhs == 0;
                let if_count = if has_else {
                    if_nodes.len() - 1
                } else {
                    if_nodes.len()
                };
                let merge_block = self.b.create_block();
                self.b.append_block_param(
                    merge_block,
                    cl_type(self.data, self.ptr_type, self.data.typ(node_id)),
                );
                self.state.block_expr_stack.push(merge_block);
                // Compile branches.
                for i in 0..if_count {
                    let condition_expr = self.compile_expr_value(if_nodes[i].lhs);
                    self.b.ins().brnz(condition_expr, then_blocks[i], &[]);
                    self.b.seal_block(then_blocks[i]);
                    if i < if_count - 1 {
                        // This is not the last else-if block.
                        let block = self.b.create_block();
                        self.b.ins().jump(block, &[]);
                        self.b.seal_block(block);
                        self.b.switch_to_block(block);
                    } else if !has_else {
                        // This is the last else-if block and there's no else.
                        self.b.ins().jump(merge_block, &[]);
                    } else {
                        // This is the last else-if block and there's an else.
                        self.b.ins().jump(then_blocks[if_count], &[]);
                        self.b.seal_block(then_blocks[if_count]);
                    }
                }
                // Compile block statements.
                for (i, if_node) in if_nodes.iter().enumerate() {
                    self.b.switch_to_block(then_blocks[i]);
                    let body = self.data.node(if_node.rhs);
                    for j in body.lhs..body.rhs {
                        let index = self.data.node_index(j);
                        self.compile_stmt(index);
                    }
                    if !self
                        .state
                        .filled_blocks
                        .contains(&self.b.current_block().unwrap())
                    {
                        self.b.ins().jump(merge_block, &[]);
                    }
                }
                self.b.seal_block(merge_block);
                self.b.switch_to_block(merge_block);
                self.state.block_expr_stack.pop();
                Val::Scalar(self.b.block_params(merge_block)[0])
            }
            Tag::Identifier => {
                let loc = self.locate(node_id);
                self.load_val(loc, node_id)
            }
            Tag::Subscript => {
                let lvalue = self.compile_lvalue(node_id);
                self.load_val(lvalue, node_id)
            }
            Tag::Conversion => {
                let x = self.compile_expr_value(node.lhs);
                let data = &self.data;
                let from_type = cl_type(data, self.ptr_type, data.typ(node.lhs));
                let to_type = cl_type(data, self.ptr_type, data.typ(node_id));
                if to_type.bytes() == from_type.bytes() {
                    Val::Scalar(x)
                } else if to_type.bytes() < from_type.bytes() {
                    Val::Scalar(self.b.ins().ireduce(to_type, x))
                } else if data.typ(node_id).is_signed() {
                    Val::Scalar(self.b.ins().sextend(to_type, x))
                } else {
                    Val::Scalar(self.b.ins().uextend(to_type, x))
                }
            }
            // Function calls
            Tag::Add | Tag::Mul => self.compile_call(node_id, node_id, true, dest),
            Tag::Call => self.compile_call(node_id, node.lhs, false, dest),
            // Arithmetic operators
            Tag::Div => compile_binary_expr!(sdiv),
            Tag::Sub => compile_binary_expr!(isub),
            Tag::Negation => {
                if self.data.tree.node(node.lhs).tag == Tag::IntegerLiteral {
                    return self.compile_integer_literal(node.lhs, true);
                }
                let lhs = self.compile_expr_value(node.lhs);
                Val::Scalar(self.b.ins().ineg(lhs))
            }
            // Bitwise operators
            Tag::BitwiseShiftL => compile_binary_expr!(ishl),
            Tag::BitwiseShiftR => compile_binary_expr!(sshr),
            Tag::BitwiseXor => compile_binary_expr!(bxor),
            // Logical operators
            Tag::Equality => compile_int_compare!(IntCC::Equal),
            Tag::Inequality => compile_int_compare!(IntCC::NotEqual),
            Tag::Greater => compile_int_compare!(IntCC::SignedGreaterThan),
            Tag::GreaterEqual => compile_int_compare!(IntCC::SignedGreaterThanOrEqual),
            Tag::Less => compile_int_compare!(IntCC::SignedLessThan),
            Tag::LessEqual => compile_int_compare!(IntCC::SignedLessThanOrEqual),
            Tag::LogicalAnd => self.compile_short_circuit(node, true),
            Tag::LogicalOr => self.compile_short_circuit(node, false),
            Tag::Not => {
                let value = self.compile_expr_value(node.lhs);
                let value = self.b.ins().icmp_imm(IntCC::Equal, value, 0);
                Val::Scalar(self.b.ins().bint(I8, value))
            }
            // Literal values
            Tag::False => Val::Scalar(self.b.ins().iconst(I8, 0)),
            Tag::True => Val::Scalar(self.b.ins().iconst(I8, 1)),
            Tag::IntegerLiteral => self.compile_integer_literal(node_id, false),
            Tag::FloatLiteral => {
                let token_str = self.data.tree.node_lexeme(node_id);
                // dbg!(token_str);
                // dbg!(data.typ(node_id));
                match self.data.typ(node_id) {
                    Typ::Numeric {
                        floating: true,
                        bytes: 4,
                        ..
                    } => {
                        let value = token_str.parse::<f32>().unwrap();
                        Val::Scalar(self.b.ins().f32const(value))
                    }
                    Typ::Numeric {
                        floating: true,
                        bytes: 8,
                        ..
                    } => {
                        let value = token_str.parse::<f64>().unwrap();
                        Val::Scalar(self.b.ins().f64const(value))
                    }
                    _ => unreachable!(),
                }
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

                // Create pointer to stored data.
                self.data_ctx.define(string.into_bytes().into_boxed_slice());
                let data_id = self
                    .state
                    .module
                    .declare_anonymous_data(true, false)
                    .unwrap();
                self.state
                    .module
                    .define_data(data_id, self.data_ctx)
                    .unwrap();
                self.data_ctx.clear();
                let local_id = self.state.module.declare_data_in_func(data_id, self.b.func);
                let data_ptr = self.b.ins().symbol_value(ty, local_id);

                let slot = if let Some(location) = dest {
                    location.stack_slot()
                } else {
                    self.create_stack_slot(node_id)
                };

                // Build string struct on stack.
                let length_value = self.b.ins().iconst(ty, length as i64);
                let layout = &Layout::new_scalar(8);
                self.store(&Location::stack(slot, 0), data_ptr, MemFlags::new(), layout);
                self.store(
                    &Location::stack(slot, 8),
                    length_value,
                    MemFlags::new(),
                    layout,
                );

                self.load_val(Location::stack(slot, 0), node_id)
            }
            _ => unreachable!("Invalid expression tag: {:?}", node.tag),
        }
    }

    fn compile_children(&mut self, node: &Node) -> (Value, Value) {
        let lhs = self.compile_expr_value(node.lhs);
        let rhs = self.compile_expr_value(node.rhs);
        (lhs, rhs)
    }

    fn compile_lvalue(&mut self, node_id: NodeId) -> Location {
        let node = self.data.tree.node(node_id);
        let flags = MemFlags::new();
        match node.tag {
            // a.x = b
            Tag::Access => self.locate_field(node_id),
            // @a = b
            Tag::Dereference => {
                // Layout of referenced variable
                let ptr_layout = &self.data.layout2(self.data.type_id(node.lhs));
                let loc = self.locate(node.lhs);
                let ptr = self.load_value(&loc, flags, self.data.typ(node.lhs), ptr_layout);
                Location::pointer(ptr, 0)
            }
            // a = b
            Tag::Identifier => self.locate_variable(node_id),
            Tag::Subscript => {
                let arr_layout = self.data.layout(node.lhs);
                let stride = if let Shape::Array { stride, .. } = arr_layout.shape {
                    stride
                } else {
                    unreachable!();
                };
                let loc = self.locate(node.lhs);
                let base = self.load_value(&loc, flags, self.data.typ(node.lhs), arr_layout);
                let index = self.compile_expr_value(node.rhs);
                let offset = self.b.ins().imul_imm(index, stride as i64);
                let addr = self.b.ins().iadd(base, offset);
                // Offset for 1-based indexing.
                Location::pointer(addr, -(stride as i32))
            }
            _ => unreachable!("Invalid lvalue {:?} for assignment", node.tag),
        }
    }

    /// If the called function is generic, the definition id refers to the generic declaration.
    /// To call the specialization, we need the right name and the right signature. At the call
    /// site, we know the base name and the arguments.
    fn compile_call(
        &mut self,
        node_id: NodeId,
        callee_id: NodeId,
        binary: bool,
        dest: Option<Location>,
    ) -> Val {
        let ty = self.ptr_type;
        let node = self.data.tree.node(node_id);
        let arg_ids = if binary {
            vec![node.lhs, node.rhs]
        } else {
            self.data
                .tree
                .range(self.data.node(node.rhs))
                .map(|i| self.data.node_index(i))
                .collect()
        };

        let mut type_arguments = vec![];
        let mut arg_type_ids = vec![];
        // let mut type_arg_type_ids = vec![];

        if !binary {
            let callee = self.data.node(node.lhs);
            for i in self.data.tree.range(callee) {
                let ni = self.data.tree.node_index(i);
                type_arguments.push(self.data.type_id(ni));
            }
        }

        for ni in &arg_ids {
            arg_type_ids.push(self.data.type_id(*ni));
        }

        let definition = self.data.definitions.get(&callee_id).unwrap_or_else(|| {
            panic!("Definition not found: {}", "failed to get function decl id")
        });

        let name = match definition {
            Definition::BuiltInFunction(id) => {
                return self.compile_built_in_function(*id, node_id);
            }
            Definition::User(id) | Definition::Overload(id) => {
                // assert_eq!(&arg_type_ids, data.typ(callee_id).parameters());
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
            Definition::Foreign(_) => self.data.tree.name(callee_id).to_string(),
            Definition::Resolved(id) => self.data.mangle_function_declaration(*id, true, &[], None),
            Definition::BuiltInType(built_in_type) => {
                let args = self.data.node(node.rhs);
                let ni = self.data.node_index(args.lhs);
                let arg = self.compile_expr_value(ni);
                let ti = *built_in_type as TypeId;
                let from_type = cl_type(self.data, self.ptr_type, self.data.typ(ni));
                let to_typ = &self.data.types[ti];
                let to_type = cl_type(self.data, self.ptr_type, to_typ);
                return if to_type.bytes() < from_type.bytes() {
                    Val::Scalar(self.b.ins().ireduce(to_type, arg))
                } else if ti == self.data.type_id(ni) {
                    Val::Scalar(arg)
                } else if to_typ.is_signed() {
                    Val::Scalar(self.b.ins().sextend(to_type, arg))
                } else {
                    Val::Scalar(self.b.ins().uextend(to_type, arg))
                };
            }
            _ => unreachable!("Definition not found: {}", "failed to get function decl id"),
        };

        let mut args: Vec<Value> = arg_ids
            .iter()
            .map(|ni| self.compile_expr_value(*ni))
            .collect();

        // let arg_types = arg_ids
        //     .iter()
        //     .map(|ni| self.data.type_id(*ni))
        //     .collect::<Vec<TypeId>>();
        let mut sig = self.state.module.make_signature();
        for ni in arg_ids {
            let t = cl_type(self.data, self.ptr_type, self.data.typ(ni));
            sig.params.push(AbiParam::new(t));
        }
        let type_id_slice = &[self.data.type_id(node_id)];
        let return_types = match self.data.typ(node_id) {
            Typ::Tuple { fields } => fields.as_slice(),
            Typ::Void => &[],
            _ => type_id_slice,
        };
        let mut return_bytes = 0;
        for ti in return_types.iter() {
            let size = sizeof(self.data.types, *ti);
            return_bytes += size;
        }
        let out_param = if return_bytes > 8 || return_types.len() > 1 {
            let location = dest.unwrap();

            let addr = if let LocationBase::Stack(slot) = location.base {
                self.b
                    .ins()
                    .stack_addr(self.ptr_type, slot, location.offset)
            } else {
                panic!();
            };

            args.push(addr);
            sig.params.push(AbiParam::special(
                self.ptr_type,
                ArgumentPurpose::StructReturn,
            ));
            Some(Val::Reference(location, Some(addr)))
        } else {
            for ti in return_types.iter() {
                sig.returns.push(AbiParam::new(cl_type(
                    self.data,
                    self.ptr_type,
                    &self.data.types[*ti],
                )))
            }
            None
        };

        let callee = self
            .state
            .module
            .declare_function(&name, Linkage::Import, &sig)
            .unwrap();
        let local_callee = self.state.module.declare_func_in_func(callee, self.b.func);

        // let cached_func = self.signatures.get(&name).unwrap();
        // assert!(sig == cached_func.1);
        // Get function data from cache
        // let callee = cached_func.0;
        // let local_callee = self
        //     .func_refs
        //     .get(&callee)
        //     .cloned()
        //     .unwrap_or_else(|| self.module.declare_func_in_func(callee, self.b.func));

        let call = self.b.ins().call(local_callee, &args);
        if let Some(ref_val) = out_param {
            return ref_val;
        }
        if !return_types.is_empty() {
            let return_values = self.b.inst_results(call);
            if return_values.len() == 1 {
                Val::Scalar(return_values[0])
            } else {
                Val::Multiple(return_values.into())
            }
        } else {
            Val::Scalar(self.b.ins().iconst(ty, 0))
        }
    }

    fn compile_built_in_function(
        &mut self,
        built_in_function: BuiltInFunction,
        node_id: NodeId,
    ) -> Val {
        let data = &self.data;
        let node = data.tree.node(node_id);
        match built_in_function {
            BuiltInFunction::Add | BuiltInFunction::AddI8 => {
                let a = self.compile_expr_value(node.lhs);
                let b = self.compile_expr_value(node.rhs);
                Val::Scalar(self.b.ins().iadd(a, b))
            }
            BuiltInFunction::Mul => {
                let a = self.compile_expr_value(node.lhs);
                let b = self.compile_expr_value(node.rhs);
                Val::Scalar(self.b.ins().imul(a, b))
            }
            BuiltInFunction::SizeOf => {
                let args = data.tree.node(node.rhs);
                let first_arg_id = data.tree.node_index(args.lhs);
                let type_id = data.type_id(first_arg_id);
                let value = sizeof(data.types, type_id) as i64;
                Val::Scalar(self.b.ins().iconst(self.ptr_type, value))
            }
            BuiltInFunction::SubI64 => {
                let a = self.compile_expr_value(node.lhs);
                let b = self.compile_expr_value(node.rhs);
                Val::Scalar(self.b.ins().isub(a, b))
            }
        }
    }

    fn compile_short_circuit(&mut self, node: &Node, is_and: bool) -> Val {
        let right_block = self.b.create_block();
        let merge_block = self.b.create_block();
        self.b.append_block_param(merge_block, I8);
        let lhs = self.compile_expr_value(node.lhs);
        if is_and {
            // If lhs is true, evaluate the rhs. Otherwise, short-circuit and jump to the merge block.
            self.b.ins().brnz(lhs, right_block, &[]);
        } else {
            // If lhs is false, evaluate the rhs. Otherwise, short-circuit and jump to the merge block.
            self.b.ins().brz(lhs, right_block, &[]);
        }
        self.b.ins().jump(merge_block, &[lhs]);
        self.b.seal_block(right_block);
        self.b.switch_to_block(right_block);
        let rhs = self.compile_expr_value(node.rhs);
        self.b.ins().jump(merge_block, &[rhs]);
        self.b.seal_block(merge_block);
        self.b.switch_to_block(merge_block);
        let value = self.b.block_params(merge_block)[0];
        let value = self.b.ins().raw_bitcast(I8, value);
        Val::Scalar(value)
    }

    fn compile_integer_literal(&mut self, node_id: NodeId, negative: bool) -> Val {
        let ty = cl_type(self.data, self.ptr_type, self.data.typ(node_id));
        let token_str = self.data.tree.node_lexeme(node_id);
        let value = token_str.parse::<i64>().unwrap();
        Val::Scalar(
            self.b
                .ins()
                .iconst(ty, if negative { -value } else { value }),
        )
    }

    fn locate(&mut self, node_id: NodeId) -> Location {
        let node = self.data.node(node_id);
        match node.tag {
            Tag::Access => self.locate_field(node_id),
            Tag::Identifier => self.locate_variable(node_id),
            _ => unreachable!("Cannot locate node with tag {:?}", node.tag),
        }
    }

    fn locate_variable(&self, node_id: NodeId) -> Location {
        let def_id = self
            .data
            .definitions
            .get_definition_id(node_id, Diagnostic::error())
            .expect("failed to lookup variable declaration");
        *self
            .state
            .locations
            .get(&def_id)
            .expect("failed to get location")
    }

    fn locate_field(&mut self, node_id: NodeId) -> Location {
        let mut indices = Vec::new();
        let mut type_ids = Vec::new();
        let mut parent_id = node_id;
        let mut parent = self.data.node(parent_id);

        let data = &self.data;

        while parent.tag == Tag::Access {
            let field_index = data
                .definitions
                .get_definition_id(parent.rhs, Diagnostic::error())
                .expect("failed to lookup field definition") as usize;
            indices.push(field_index);
            parent_id = parent.lhs;
            if let Typ::Pointer { typ, .. } = data.typ(parent_id) {
                type_ids.push(*typ)
            } else {
                type_ids.push(data.type_id(parent_id))
            }
            parent = data.node(parent_id);
        }

        let mut offset = 0;
        for i in (0..indices.len()).rev() {
            let layout = &data.layouts[type_ids[i]];
            offset += layout.shape.offset(indices[i]);
        }

        let mut location = self.locate_variable(parent_id);
        // Dereference pointer values.
        if let Typ::Pointer { .. } = &data.typ(parent_id) {
            let ptr_value = self.load_value(
                &location,
                MemFlags::new(),
                data.typ(parent_id),
                data.layout(parent_id),
            );
            location = Location::pointer(ptr_value, 0)
        }
        location.offset(offset)
    }

    fn create_stack_slot(&mut self, node_id: u32) -> StackSlot {
        let data = &self.data;
        let type_id = if let Typ::Parameter { binding, .. } = data.typ(node_id) {
            *binding
        } else {
            data.type_id(node_id)
        };
        let size = data.sizeof(type_id);
        let size = next_power_of_16(size);
        self.b
            .create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size))
    }

    fn get_addr(&mut self, location: &Location) -> Value {
        let ins = self.b.ins();
        match location.base {
            LocationBase::Pointer(base_addr) => {
                let offset = location.offset as i64;
                if offset == 0 {
                    base_addr
                } else {
                    ins.iadd_imm(base_addr, offset)
                }
            }
            LocationBase::Stack(stack_slot) => {
                ins.stack_addr(self.ptr_type, stack_slot, location.offset)
            }
            LocationBase::Register(_) => unreachable!("cannot get address of register"),
        }
    }

    fn load_scalar(&mut self, location: &Location, flags: MemFlags, typ: &Typ) -> Value {
        let ins = self.b.ins();
        let ty = cl_type(self.data, self.ptr_type, typ);
        match location.base {
            LocationBase::Pointer(base_addr) => ins.load(ty, flags, base_addr, location.offset),
            LocationBase::Register(variable) => self.b.use_var(variable),
            LocationBase::Stack(stack_slot) => ins.stack_load(ty, stack_slot, location.offset),
        }
    }

    fn load_value(
        &mut self,
        location: &Location,
        flags: MemFlags,
        typ: &Typ,
        layout: &Layout,
    ) -> Value {
        let (ins, ty) = (self.b.ins(), self.ptr_type);
        match location.base {
            LocationBase::Pointer(_) | LocationBase::Register(_) => {
                self.load_scalar(location, flags, typ)
            }
            LocationBase::Stack(stack_slot) => {
                if layout.size <= 8 {
                    self.load_scalar(location, flags, typ)
                } else {
                    ins.stack_addr(ty, stack_slot, location.offset)
                }
            }
        }
    }

    fn load_val(&mut self, location: Location, node_id: NodeId) -> Val {
        let ins = self.b.ins();
        let typ = self.data.typ(node_id);
        let layout = self.data.layout(node_id);
        match location.base {
            LocationBase::Register(variable) => Val::Scalar(self.b.use_var(variable)),
            LocationBase::Pointer(_) => Val::Reference(location, None),
            LocationBase::Stack(stack_slot) => {
                if layout.size <= 8 {
                    let ty = cl_type(self.data, self.ptr_type, typ);
                    Val::Scalar(ins.stack_load(ty, stack_slot, location.offset))
                } else {
                    Val::Reference(location, None)
                }
            }
        }
    }

    fn store(&mut self, location: &Location, value: Value, flags: MemFlags, layout: &Layout) {
        let (ins, ty) = (self.b.ins(), self.ptr_type);
        match location.base {
            // lvalue is a pointer, store in address, rvalue is scalar/aggregate
            LocationBase::Pointer(base_addr) => {
                if layout.size <= 8 {
                    ins.store(flags, value, base_addr, location.offset);
                } else {
                    let size = next_power_of_16(layout.size) as u64;
                    let align = layout.align as u8;
                    let src_addr = value;
                    let offset = self.b.ins().iconst(self.ptr_type, location.offset as i64);
                    let dest_addr = self.b.ins().iadd(base_addr, offset);
                    dbg!(align, layout.size);
                    self.b.emit_small_memory_copy(
                        self.target_config,
                        dest_addr,
                        src_addr,
                        size,
                        align,
                        align,
                        true,
                        MemFlags::new(),
                    );
                }
            }
            // lvalue is a register variable, rvalue must be a scalar
            LocationBase::Register(variable) => self.b.def_var(variable, value),
            // lvalue is a stack variable, rvalue is scalar/aggregate
            LocationBase::Stack(stack_slot) => {
                if layout.size <= 8 {
                    ins.stack_store(value, stack_slot, location.offset);
                } else {
                    let size = next_power_of_16(layout.size) as u64;
                    let align = layout.align as u8;
                    // let align = dbg!((layout.size as i64 & -(layout.size as i64)) as u8);
                    let src_addr = value;
                    let dest_addr = ins.stack_addr(ty, stack_slot, location.offset);
                    self.b.emit_small_memory_copy(
                        self.target_config,
                        dest_addr,
                        src_addr,
                        size,
                        align,
                        align,
                        true,
                        MemFlags::new(),
                    );
                }
            }
        };
    }

    // fn store_expr_in_memory(
    //     &mut self,
    //     node_id: NodeId,
    //     type_id: TypeId,
    //     stack_slot: StackSlot,
    //     stack_addr: Value,
    //     offset: u32,
    // ) {
    //     if struct_info.is_none() {
    //         if let Some(fields) = expr_ty.as_struct() {
    //             let fields = fields.into_iter().map(|(_, ty)| ty).collect::<Vec<_>>();
    //             let struct_mem = StructMemory::new(fields.clone(), self.pointer_ty);
    //             struct_info.replace((fields, struct_mem));
    //         }
    //     }
    //     let node = self.data.node(node_id);

    //     match node. {
    //         hir::Expr::Array {
    //             items: Some(items), ..
    //         } => self.store_array_items(items.clone(), stack_slot, stack_addr, offset),
    //         hir::Expr::StructLiteral {
    //             fields: field_values,
    //             ..
    //         } => {
    //             let field_tys = struct_info.as_ref().unwrap().0.clone();
    //             let struct_mem = &struct_info.as_ref().unwrap().1;

    //             self.store_struct_fields(
    //                 struct_mem,
    //                 field_tys,
    //                 field_values.iter().map(|(_, val)| *val).collect(),
    //                 stack_slot,
    //                 stack_addr,
    //                 offset,
    //             )
    //         }
    //         _ if expr_ty.is_aggregate() => {
    //             let far_off_thing = self.compile_expr(expr).unwrap();

    //             let offset = self.builder.ins().iconst(self.pointer_ty, offset as i64);

    //             let actual_addr = self.builder.ins().iadd(stack_addr, offset);

    //             let size = self.builder.ins().iconst(self.pointer_ty, expr_size as i64);

    //             self.builder.call_memcpy(
    //                 self.module.target_config(),
    //                 actual_addr,
    //                 far_off_thing,
    //                 size,
    //             )
    //         }
    //         _ => {
    //             if let Some(item) = self.compile_expr(expr) {
    //                 self.builder
    //                     .ins()
    //                     .stack_store(item, stack_slot, offset as i32);
    //             }
    //         }
    //     }
    // }

    fn cranelift_value(&mut self, val: Val, node_id: NodeId) -> Value {
        let typ = self.data.typ(node_id);
        let layout = self.data.layout(node_id);
        match &val {
            Val::Scalar(value) => *value,
            // location is a pointer or stack slot
            Val::Reference(location, _) => {
                if layout.size <= 8 {
                    self.load_scalar(location, MemFlags::new(), typ)
                } else {
                    self.get_addr(location)
                }
            }
            _ => unreachable!("cannot get cranelift_value for multiple"),
        }
    }
}

pub struct State {
    locations: HashMap<NodeId, Location>,
    pub module: Box<dyn CraneliftModule>,
    filled_blocks: HashSet<Block>,
    block_expr_stack: Vec<Block>,
    pub signatures: HashMap<String, (FuncId, Signature)>,
    // func_refs: HashMap<FuncId, FuncRef>,
}

pub fn cl_type(data: &Data, ptr_type: Type, typ: &Typ) -> Type {
    cl_type_with(data, ptr_type, typ, data.active_type_parameters.last())
}

pub fn cl_type_with(
    data: &Data,
    ptr_type: Type,
    typ: &Typ,
    type_arguments: Option<&&Vec<TypeId>>,
) -> Type {
    match typ {
        Typ::Boolean => I8,
        Typ::Pointer { .. } => ptr_type,
        Typ::Numeric {
            floating, bytes, ..
        } => {
            if *floating {
                match bytes {
                    4 => F32,
                    8 => F64,
                    _ => unreachable!("Float types must be 4 or 8 bytes."),
                }
            } else {
                match bytes {
                    1 => I8,
                    2 => I16,
                    4 => I32,
                    8 => I64,
                    _ => unreachable!("Integer types must be 1/2/4/8 bytes."),
                }
            }
        }
        Typ::TypeParameter { index, .. } => {
            cl_type(data, ptr_type, &data.types[type_arguments.unwrap()[*index]])
        }
        Typ::Parameter { binding, .. } => cl_type(data, ptr_type, &data.types[*binding]),
        _ => ptr_type,
        // _ => unreachable!("Invalid type: {typ:?} is not a primitive Cranelift type."),
    }
}

// https://github.com/bjorn3/rustc_codegen_cranelift/blob/master/src/pointer.rs
#[derive(Copy, Clone, Debug)]
pub struct Location {
    base: LocationBase,
    offset: i32,
}

#[allow(dead_code)]
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

    fn offset(self, extra_offset: i32) -> Self {
        Location {
            base: self.base,
            offset: self.offset + extra_offset,
        }
    }

    fn stack_slot(self) -> StackSlot {
        if let LocationBase::Stack(stack_slot) = self.base {
            stack_slot
        } else {
            unreachable!("can only get the stack slot for a stack location");
        }
    }
}

#[derive(Debug, Clone)]
pub enum Val {
    Scalar(Value),
    Reference(Location, Option<Value>),
    Multiple(Vec<Value>),
}

fn is_struct_return(data: &Data, return_types: &[TypeId]) -> (bool, u32) {
    let mut return_bytes = 0;
    for ti in return_types.iter() {
        let size = sizeof(data.types, *ti);
        return_bytes += size;
    }
    (return_types.len() > 1 || return_bytes > 8, return_bytes)
}

fn next_power_of_16(x: u32) -> u32 {
    (x + 15) & (!16 + 1)
}
