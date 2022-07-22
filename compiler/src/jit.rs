use crate::parse::NodeId;
use core::mem;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::analyze::Lookup;
use crate::parse::{Node, Tag, Tree};
use crate::typecheck::Type as Typ;
use cranelift::codegen;
use cranelift::prelude::isa;
use cranelift::prelude::settings;
use cranelift::prelude::{
    AbiParam, FunctionBuilder, FunctionBuilderContext, InstBuilder, IntCC, StackSlotData,
    StackSlotKind, Type, Value, Variable,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

#[derive(Debug, Clone)]
pub enum Val {
    Scalar(Value),
    Aggregate(Vec<Val>),
}

impl Val {
    pub fn values(self) -> Vec<Value> {
        let mut values = Vec::new();
        self.push_contents(&mut values);
        values
    }

    fn push_contents(self, values: &mut Vec<Value>) {
        match self {
            Val::Scalar(value) => {
                values.push(value);
            }
            Val::Aggregate(vals) => {
                for val in vals {
                    val.push_contents(values);
                }
            }
        }
    }

    pub fn value(self) -> Value {
        if let Val::Scalar(value) = self {
            value
        } else {
            panic!("Value::Struct found in eval")
        }
    }
}

#[derive(Debug, Clone)]
pub enum Var {
    Scalar(Variable),
    Aggregate(Vec<Var>),
}

impl Var {
    pub fn variables(self) -> Vec<Variable> {
        let mut variables = Vec::new();
        self.push_contents(&mut variables);
        variables
    }

    fn push_contents(self, variables: &mut Vec<Variable>) {
        match self {
            Var::Scalar(variable) => {
                variables.push(variable);
            }
            Var::Aggregate(vars) => {
                for var in vars {
                    var.push_contents(variables);
                }
            }
        }
    }

    pub fn variable(&self) -> Variable {
        if let Var::Scalar(value) = self {
            *value
        } else {
            panic!("Variable::Struct found in eval")
        }
    }
}

#[test]
fn test_val_values() {
    let v0 = Val::Scalar(Value::from_u32(0));
    let v1 = Val::Scalar(Value::from_u32(1));
    let val = Val::Aggregate(vec![v0, v1]);
    assert_eq!(val.values().len(), 2);
    let v0 = Val::Scalar(Value::from_u32(0));
    let v1 = Val::Scalar(Value::from_u32(1));
    let v2 = Val::Scalar(Value::from_u32(2));
    let v3 = Val::Scalar(Value::from_u32(3));
    let val = Val::Aggregate(vec![
        v0,
        Val::Aggregate(vec![v1, Val::Aggregate(vec![v2]), v3]),
    ]);
    assert_eq!(val.values().len(), 4);
}

pub trait CraneliftModule: Module {
    fn finalize(self: Box<Self>, id: FuncId, output_file: &Path);
}

impl CraneliftModule for JITModule {
    fn finalize(mut self: Box<Self>, id: FuncId, _output_file: &Path) {
        self.finalize_definitions();
        let main = self.get_finalized_function(id);
        // let main: fn(i64, i64) -> i64 = unsafe { mem::transmute(main) };
        // let result = main(3, 4);
        let main: fn() -> i64 = unsafe { mem::transmute(main) };
        let result = main();
        // let result = code_fn(input, input);
        println!("JIT result: {}", result);
    }
}

impl CraneliftModule for ObjectModule {
    fn finalize(self: Box<Self>, _id: FuncId, output_file: &Path) {
        let product = self.finish();
        let bytes = product.emit().unwrap();
        fs::write(output_file, bytes).unwrap();
    }
}

pub struct Generator<'a> {
    pub builder_ctx: FunctionBuilderContext,
    pub ctx: codegen::Context,
    data_ctx: DataContext,
    pub module: Box<dyn CraneliftModule>,
    input: &'a Input<'a>,
    state: State,
}

pub struct State {
    // scope: usize,
    // latest_scope: usize,
    var_index: u32,
    variables: HashMap<u32, Var>,
    stack_slots: HashMap<u32, codegen::ir::StackSlot>,
}

impl State {
    ///
    pub fn compile_stmt(
        &mut self,
        input: &Input,
        module: &mut Box<dyn CraneliftModule>,
        builder: &mut FunctionBuilder,
        index: u32,
        ty: Type,
    ) {
        let node = input.node(index);
        match node.tag {
            Tag::Assign => {
                // lhs: expr
                // rhs: expr
                println!("Compiling assign statment.");
                let val = self
                    .compile_expr(input, module, builder, node.rhs, ty)
                    .value();
                let lhs = input.node(node.lhs);
                match lhs.tag {
                    // a = b
                    Tag::Identifier => {
                        let var = self.lookup_var(input, node.lhs);
                        builder.def_var(var.variable(), val);
                    }
                    // a.x = b
                    Tag::Access => {
                        if let Lookup::Defined(decl_id) = input.definitions.get(&lhs.lhs).unwrap() {
                            let slot = *self.stack_slots.get(&decl_id).unwrap();
                            let field_index = if let Lookup::Defined(fi) =
                                input.definitions.get(&lhs.rhs).unwrap()
                            {
                                println!("field_index: {}", fi);
                                *fi
                            } else {
                                0
                            };
                            let offset = (field_index * ty.bytes()) as i32;
                            // builder.ins().stack_store(val, slot, offset);
                            let variables = self.lookup_var(input, lhs.lhs).variables();
                            builder.def_var(variables[field_index as usize], val);
                        }
                        // let def = self.definitions
                        //     builder.ins().store(MemFlags::new(), val)
                    }
                    _ => unreachable!(),
                };
            }
            Tag::If => {
                let condition_expr = self
                    .compile_expr(input, module, builder, node.lhs, ty)
                    .value();
                let then_block = builder.create_block();
                let merge_block = builder.create_block();
                let body = input.node(node.rhs);
                builder.ins().brz(condition_expr, merge_block, &[]);
                builder.ins().jump(then_block, &[]);
                builder.seal_block(then_block);
                // then block
                builder.switch_to_block(then_block);
                for i in body.lhs..body.rhs {
                    let index = input.tree.node_index(i);
                    self.compile_stmt(input, module, builder, index, ty);
                }
                if !builder.is_filled() {
                    builder.ins().jump(merge_block, &[]);
                }
                builder.seal_block(merge_block);
                // merge block
                builder.switch_to_block(merge_block);
            }
            Tag::IfElse => {
                let mut if_nodes = Vec::new();
                let mut then_blocks = Vec::new();
                for i in node.lhs..node.rhs {
                    let index = input.node_index(i);
                    let if_node = input.node(index);
                    if_nodes.push(if_node);
                    then_blocks.push(builder.create_block());
                }
                let if_count = if if_nodes.last().unwrap().lhs == 0 {
                    if_nodes.len() - 1
                } else {
                    if_nodes.len()
                };
                let merge_block = builder.create_block();
                for i in 0..if_count {
                    let condition_expr = self
                        .compile_expr(input, module, builder, if_nodes[i].lhs, ty)
                        .value();
                    builder.ins().brnz(condition_expr, then_blocks[i], &[]);
                    builder.seal_block(then_blocks[i]);
                    if i < if_count - 1 {
                        let block = builder.create_block();
                        builder.ins().jump(block, &[]);
                        builder.seal_block(block);
                        builder.switch_to_block(block);
                    }
                    if i == if_nodes.len() - 1 {
                        builder.ins().jump(merge_block, &[]);
                    }
                }
                for i in if_count..if_nodes.len() {
                    builder.ins().jump(then_blocks[i], &[]);
                    builder.seal_block(then_blocks[i]);
                }
                for (i, if_node) in if_nodes.iter().enumerate() {
                    builder.switch_to_block(then_blocks[i]);
                    let body = input.node(if_node.rhs);
                    for j in body.lhs..body.rhs {
                        let index = input.node_index(j);
                        self.compile_stmt(input, module, builder, index, ty);
                    }
                    builder.ins().jump(merge_block, &[]);
                }
                builder.seal_block(merge_block);
                builder.switch_to_block(merge_block);
            }
            Tag::VariableDecl => {
                // lhs: type
                // rhs: expr
                println!("Compiling variable declaration statement.");
                if node.lhs != 0 && node.rhs == 0 {
                    if let Lookup::Defined(def_index) = input.definitions.get(&node.lhs).unwrap() {
                        // Tag::Struct or Tag::VariableDecl
                        let def_node = input.node(*def_index);
                        if def_node.tag == Tag::Struct {
                            let struct_size = Self::get_struct_size(input, def_node);
                            // println!("{:?} {:?}", def_index, def_node.rhs - def_node.lhs);
                            let slot = builder.create_stack_slot(StackSlotData::new(
                                StackSlotKind::ExplicitSlot,
                                struct_size,
                            ));
                            // Map VariableDecl id to slot
                            self.stack_slots.insert(index, slot);
                            println!("{}", struct_size);
                        }
                    } else {
                        panic!("Undefined type")
                    }
                }
                // let var = self.create_var(index).variable();
                // builder.declare_var(var, ty);
                // let val = if node.rhs != 0 {
                //     self.compile_expr(input, module, builder, node.rhs, ty)
                //         .value()
                // } else {
                //     builder.ins().iconst(ty, 0)
                // };
                // builder.def_var(var, val);

                let var = self.create_struct_var(input, index);
                dbg!(&var);
                for variable in var.variables() {
                    builder.declare_var(variable, ty);
                    let val = if node.rhs != 0 {
                        self.compile_expr(input, module, builder, node.rhs, ty)
                            .value()
                    } else {
                        builder.ins().iconst(ty, 0)
                    };
                    builder.def_var(variable, val);
                }
            }
            Tag::Return => {
                println!("Return");
                let val = self
                    .compile_expr(input, module, builder, node.lhs, ty)
                    .value();
                builder.ins().return_(&[val]);
            }
            Tag::While => {
                let condition = self
                    .compile_expr(input, module, builder, node.lhs, ty)
                    .value();
                let while_block = builder.create_block();
                let merge_block = builder.create_block();
                // check condition
                // true? jump to loop body
                builder.ins().brnz(condition, while_block, &[]);
                // false? jump to after loop
                builder.ins().jump(merge_block, &[]);
                // block_while:
                builder.switch_to_block(while_block);
                let body = input.node(node.rhs);
                for i in body.lhs..body.rhs {
                    let index = input.node_index(i);
                    self.compile_stmt(input, module, builder, index, ty);
                }
                let condition = self
                    .compile_expr(input, module, builder, node.lhs, ty)
                    .value();
                // brnz block_while
                builder.ins().brnz(condition, while_block, &[]);
                builder.seal_block(while_block);
                builder.ins().jump(merge_block, &[]);
                builder.seal_block(merge_block);
                // block_merge:
                builder.switch_to_block(merge_block);
            }
            _ => {
                self.compile_expr(input, module, builder, index, ty);
            }
        }
    }

    fn get_struct_size(input: &Input, node: &Node) -> u32 {
        let mut struct_size = 0;
        for i in node.lhs..node.rhs {
            let ni = input.node_index(i);
            // Tag::Field
            let ti = input.node_types[ni as usize];
            println!(
                "{}: {:?} ({} bytes)",
                input.tree.node_lexeme(ni),
                input.types[ti],
                input.sizeof(ti),
            );
            struct_size += input.sizeof(ti);
        }
        struct_size
    }

    ///
    pub fn compile_expr(
        &mut self,
        input: &Input,
        module: &mut Box<dyn CraneliftModule>,
        builder: &mut FunctionBuilder,
        index: NodeId,
        ty: Type,
    ) -> Val {
        let node = input.node(index);
        // println!("expr: {:?}", node.tag);
        dbg!(node.tag);
        match node.tag {
            Tag::Access => {
                println!(
                    "Compiling access expr: {}.{}.",
                    input.tree.node_lexeme(node.lhs),
                    input.tree.node_lexeme(node.rhs)
                );
                // Get the struct stack variable.
                let struct_type = input.node_types[node.lhs as usize];
                if let Lookup::Defined(decl_id) = input.definitions.get(&node.lhs).unwrap() {
                    // let slot = *self.stack_slots.get(&decl_id).unwrap();
                    let field_index =
                        if let Lookup::Defined(fi) = input.definitions.get(&node.rhs).unwrap() {
                            *fi
                        } else {
                            0
                        };
                    let offset = (field_index * ty.bytes()) as i32;
                    dbg!(decl_id);
                    let var = self.lookup_var(input, node.lhs);
                    dbg!(&var);
                    let struct_variables = var.variables();
                    let value = builder.use_var(struct_variables[field_index as usize]);
                    dbg!(&struct_variables, &field_index, &value);
                    Val::Scalar(value)
                    // Val::Scalar(builder.ins().stack_load(ty, slot, offset))
                } else {
                    Val::Scalar(builder.ins().iconst(ty, 0))
                }
            }
            // Tag::Identifier => {
            //     let var = self.lookup_var(input, index);
            //     Val::Scalar(builder.use_var(var))
            // }
            Tag::Add => {
                let lhs = self
                    .compile_expr(input, module, builder, node.lhs, ty)
                    .value();
                let rhs = self
                    .compile_expr(input, module, builder, node.rhs, ty)
                    .value();
                dbg!(&lhs, &rhs);
                Val::Scalar(builder.ins().iadd(lhs, rhs))
            }
            Tag::Sub => {
                let lhs = self
                    .compile_expr(input, module, builder, node.lhs, ty)
                    .value();
                let rhs = self
                    .compile_expr(input, module, builder, node.rhs, ty)
                    .value();
                Val::Scalar(builder.ins().isub(lhs, rhs))
            }
            Tag::Div => {
                let lhs = self
                    .compile_expr(input, module, builder, node.lhs, ty)
                    .value();
                let rhs = self
                    .compile_expr(input, module, builder, node.rhs, ty)
                    .value();
                Val::Scalar(builder.ins().sdiv(lhs, rhs))
            }
            Tag::Mul => {
                let lhs = self
                    .compile_expr(input, module, builder, node.lhs, ty)
                    .value();
                let rhs = self
                    .compile_expr(input, module, builder, node.rhs, ty)
                    .value();
                Val::Scalar(builder.ins().imul(lhs, rhs))
            }
            Tag::Greater => {
                let lhs = self
                    .compile_expr(input, module, builder, node.lhs, ty)
                    .value();
                let rhs = self
                    .compile_expr(input, module, builder, node.rhs, ty)
                    .value();
                Val::Scalar(builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs))
            }
            Tag::Less => {
                let lhs = self
                    .compile_expr(input, module, builder, node.lhs, ty)
                    .value();
                let rhs = self
                    .compile_expr(input, module, builder, node.rhs, ty)
                    .value();
                Val::Scalar(builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs))
            }
            Tag::Grouping => self.compile_expr(input, module, builder, node.lhs, ty),
            Tag::IntegerLiteral => {
                let token_str = input.node_lexeme_offset(node, 0);
                let value = token_str.parse::<i64>().unwrap();
                Val::Scalar(builder.ins().iconst(ty, value))
            }
            Tag::Call => {
                let mut sig = module.make_signature();

                let function = input.node(node.lhs);
                let name = input.node_lexeme_offset(function, 0);
                println!("{}", name);
                let arguments = input.node(node.rhs);
                let mut args = Vec::new();
                for _ in arguments.lhs..arguments.rhs {
                    println!("argument");
                    // sig.params.push(AbiParam::new(ty));
                }
                for i in arguments.lhs..arguments.rhs {
                    let index = input.node_index(i);
                    let mut values = self
                        .compile_expr(input, module, builder, index, ty)
                        .values();
                    for _ in &values {
                        sig.params.push(AbiParam::new(ty));
                    }
                    args.append(&mut values);
                }
                // Assume one return value.
                sig.returns.push(AbiParam::new(ty));
                println!("{}", sig);
                let callee = module
                    .declare_function(name, Linkage::Import, &sig)
                    .unwrap();
                let local_callee = module.declare_func_in_func(callee, builder.func);

                let call = builder.ins().call(local_callee, &args);
                Val::Scalar(builder.inst_results(call)[0])
                // let mut sig = return builder.ins().iconst(ty, 20);
            }
            Tag::Identifier => {
                let var = self.lookup_var(input, index);
                let val = Self::var_to_val(builder, var);
                dbg!(&val);
                val
                // Val::Scalar(builder.use_var(var.variable()))
            }
            _ => Val::Scalar(builder.ins().iconst(ty, 0)),
        }
    }

    fn var_to_val(builder: &mut FunctionBuilder, var: Var) -> Val {
        match var {
            Var::Scalar(variable) => Val::Scalar(builder.use_var(variable)),
            Var::Aggregate(vars) => {
                let mut vals = Vec::new();
                for var in vars {
                    vals.push(Self::var_to_val(builder, var));
                }
                Val::Aggregate(vals)
            }
        }
    }

    /// Maps node_id -> Var.
    fn lookup_var(&self, input: &Input, node_id: u32) -> Var {
        println!("lookup_var: \"{}\"", node_id);
        if let Lookup::Defined(decl_id) = input.definitions.get(&node_id).unwrap() {
            println!("decl_id: \"{}\"", decl_id);
            let var = self.variables.get(&decl_id).unwrap().clone();
            var
        } else {
            panic!();
        }
    }

    ///
    fn create_var(&mut self, node_id: u32) -> Var {
        let var = Var::Scalar(Variable::with_u32(self.var_index));
        self.variables.insert(node_id, var.clone());
        self.var_index += 1;
        var
    }

    ///
    fn create_struct_var(&mut self, input: &Input, node_id: u32) -> Var {
        // let v = Variable::with_u32(self.var_index);
        // let t = input.node_type(node_id);
        println!("create_struct_var: \"{}\"", node_id);
        let var = self.node_type_to_var(input, node_id);
        self.variables.insert(node_id, var.clone());
        var
    }

    fn node_type_to_var(&mut self, input: &Input, node_id: NodeId) -> Var {
        let type_id = input.node_types[node_id as usize];
        self.type_to_var(input, type_id)
    }

    fn type_to_var(&mut self, input: &Input, type_id: usize) -> Var {
        let t = &input.types[type_id];
        dbg!(t);
        match t {
            Typ::Struct { fields } => {
                let mut vars = Vec::new();
                for type_id in fields {
                    let var = self.type_to_var(input, *type_id);
                    vars.push(var);
                }
                Var::Aggregate(vars)
            }
            _ => {
                let variable = Variable::with_u32(self.var_index);
                self.var_index += 1;
                Var::Scalar(variable)
            }
        }
    }

    fn push_variables(&mut self, input: &Input, type_id: usize, vars: &mut Vec<Var>) {
        let t = &input.types[type_id];
        match t {
            Typ::Struct { fields } => {
                for type_id in fields {
                    self.push_variables(input, *type_id, vars);
                }
            }
            _ => {
                vars.push(Var::Scalar(Variable::with_u32(self.var_index)));
                self.var_index += 1;
            }
        }
    }
}

pub struct Input<'a> {
    pub tree: &'a Tree,
    pub definitions: &'a HashMap<u32, Lookup>,
    pub types: &'a Vec<Typ>,
    pub node_types: &'a Vec<usize>,
}

impl<'a> Input<'a> {
    fn node(&self, index: u32) -> &Node {
        self.tree.node(index)
    }
    fn node_index(&self, index: u32) -> u32 {
        self.tree.node_index(index)
    }

    fn node_indirect(&self, index: u32) -> &Node {
        self.tree.node_indirect(index)
    }
    fn node_lexeme_offset(&self, node: &Node, offset: i32) -> &str {
        self.tree.node_lexeme_offset(node, offset)
    }

    pub fn sizeof(&self, type_id: usize) -> u32 {
        match &self.types[type_id] {
            Typ::Void => 0,
            Typ::Array { .. } => 16,
            Typ::Struct { fields } => {
                let mut size = 0;
                for f in fields {
                    size += self.sizeof(*f);
                }
                size
            }
            _ => 8,
        }
    }

    pub fn node_type(&self, node_id: NodeId) -> &Typ {
        &self.types[self.node_types[node_id as usize]]
    }
}

impl<'a> Generator<'a> {
    pub fn new(input: &'a Input<'a>, output_name: String, use_jit: bool) -> Self {
        // let module = CraneliftModule::new(output_name, use_jit);
        let flag_builder = settings::builder();
        // flag_builder.enable("is_pic").unwrap();
        let isa_builder = isa::lookup(target_lexicon::HOST).unwrap();
        println!("{}", target_lexicon::HOST);
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();

        let module: Box<dyn CraneliftModule> = if use_jit {
            let jit_builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
            Box::new(JITModule::new(jit_builder))
        } else {
            let object_builder =
                ObjectBuilder::new(isa, output_name, cranelift_module::default_libcall_names())
                    .unwrap();
            Box::new(ObjectModule::new(object_builder))
        };
        //
        Self {
            builder_ctx: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_ctx: DataContext::new(),
            module,
            input,
            state: State {
                var_index: 0,
                variables: HashMap::new(),
                stack_slots: HashMap::new(),
            },
        }
    }

    ///
    pub fn init_builder<'b>(
        func: &'b mut codegen::ir::Function,
        builder_ctx: &'b mut FunctionBuilderContext,
        input: &'b Input,
        parameters: &Node,
        returns: &Node,
        t: Type,
    ) -> FunctionBuilder<'b> {
        for i in parameters.lhs..parameters.rhs {
            // Tag::Field
            let ni = input.node_index(i);
            Self::push_params(input, ni, &mut func.signature.params, t);
            // func.signature.params.push(AbiParam::new(t));
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
        FunctionBuilder::new(func, builder_ctx)
    }

    fn push_params(input: &Input, node_id: NodeId, params: &mut Vec<AbiParam>, t: Type) {
        let field_node = input.node(node_id);
        if let Lookup::Defined(decl_id) = input.definitions.get(&field_node.lhs).unwrap() {
            let decl_node = input.node(*decl_id);
            if decl_node.tag == Tag::Struct {
                // Defined struct
                for i in decl_node.lhs..decl_node.rhs {
                    let field_id = input.node_index(i);
                    Self::push_params(input, field_id, params, t);
                }
                return;
            }
        }
        // Defined or built-in primitive
        params.push(AbiParam::new(t));
    }

    ///
    pub fn compile_nodes(mut self, filename: &Path) {
        let root = &self.input.tree.nodes[0];
        // println!("{:?}", root.tag);
        let mut func_ids = Vec::new();
        let mut main_id = None;
        for i in root.lhs..root.rhs {
            let module_index = self.input.node_index(i);
            let module = self.input.node(module_index);

            if module.tag != Tag::Module {
                continue;
            }

            for i in module.lhs..module.rhs {
                let ni = self.input.node_index(i);
                let node = self.input.node(ni);
                match node.tag {
                    Tag::FunctionDecl => {
                        let func_id = self.compile_function_decl(ni);
                        let func_name = self.input.tree.node_lexeme_offset(&node, -1);
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
            self.module.finalize(id, filename);
        }
        println!("Finalized {:?}", func_ids);
    }

    ///
    pub fn compile_function_decl(&mut self, index: u32) -> FuncId {
        dbg!("compile_function_decl");
        let int = self.module.target_config().pointer_type();
        let node = self.input.node(index);
        dbg!(self.input.node_lexeme_offset(node, -1));
        assert_eq!(node.tag, Tag::FunctionDecl);
        let prototype = self.input.node(node.lhs);
        assert_eq!(prototype.tag, Tag::Prototype);
        let parameters = self.input.node(prototype.lhs);
        assert_eq!(parameters.tag, Tag::Parameters);
        let returns = self.input.node(prototype.rhs);
        let mut builder = Self::init_builder(
            &mut self.ctx.func,
            &mut self.builder_ctx,
            self.input,
            parameters,
            returns,
            int,
        );
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        dbg!(builder.block_params(entry_block).len());

        // Define parameters as local variables.
        let mut scalar_count = 0;
        for i in parameters.lhs..parameters.rhs {
            let ni = self.input.node_index(i);
            let var = self.state.create_struct_var(self.input, ni);
            dbg!(&var);
            for &variable in &var.variables() {
                builder.declare_var(variable, int);
                let val = builder.block_params(entry_block)[scalar_count];
                builder.def_var(variable, val);
                scalar_count += 1;
            }
        }

        let body = self.input.node(node.rhs);
        for i in body.lhs..body.rhs {
            let index = self.input.node_index(i);
            self.state
                .compile_stmt(&self.input, &mut self.module, &mut builder, index, int);
        }

        builder.finalize();
        let name = self.input.tree.node_lexeme_offset(node, -1);
        println!("{} {}", name, builder.func.display());
        let id = self
            .module
            .declare_function(name, Linkage::Export, &self.ctx.func.signature)
            .unwrap();
        self.module.define_function(id, &mut self.ctx).unwrap();
        self.module.clear_context(&mut self.ctx);
        id
    }
}
