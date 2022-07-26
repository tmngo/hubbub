use crate::parse::NodeId;
use core::mem;
use cranelift::prelude::MemFlags;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::analyze::Definition;
use crate::parse::{Node, Tag, Tree};
use crate::typecheck::{Type as Typ, TypeId, TypeIndex};
use cranelift::codegen;
use cranelift::codegen::ir::StackSlot;
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
        let mut values = Vec::<Value>::new();
        let mut stack = vec![&self];
        loop {
            match stack.pop() {
                Some(val) => match val {
                    Val::Scalar(value) => values.push(*value),
                    Val::Aggregate(vals) => stack.extend(vals),
                },
                None => return values,
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

#[derive(Debug, Clone)]
pub enum Var {
    Scalar(Variable),
    Aggregate(Vec<Var>),
}

impl Var {
    pub fn variables(self) -> Vec<Variable> {
        let mut variables = Vec::new();
        let mut stack = vec![&self];
        loop {
            match stack.pop() {
                Some(val) => match val {
                    Var::Scalar(variable) => variables.push(*variable),
                    Var::Aggregate(vars) => stack.extend(vars),
                },
                None => return variables,
            }
        }
    }

    pub fn variable(&self) -> Variable {
        if let Var::Scalar(value) = self {
            return *value;
        }
        panic!("expected scalar variable, got aggregate")
    }

    fn to_val(&self, builder: &mut FunctionBuilder) -> Val {
        match self {
            Var::Scalar(variable) => Val::Scalar(builder.use_var(*variable)),
            Var::Aggregate(vars) => {
                let mut vals = Vec::new();
                for var in vars {
                    vals.push(var.to_val(builder));
                }
                Val::Aggregate(vals)
            }
        }
    }

    fn set_value(&self, b: &mut FunctionBuilder, val: Value) {
        b.def_var(self.variable(), val);
    }

    fn set_field(&self, b: &mut FunctionBuilder, val: Value, field_index: u32) {
        let variables = self.clone().variables();
        b.def_var(variables[field_index as usize], val);
    }
}

#[derive(Debug, Clone)]
pub struct StackVar {
    slot: StackSlot,
    offsets: Vec<i32>,
}

impl StackVar {
    fn set_value(&self, b: &mut FunctionBuilder, val: Value) {
        b.ins().stack_store(val, self.slot, 0);
    }

    fn set_field(&self, b: &mut FunctionBuilder, val: Value, field_index: u32) {
        b.ins()
            .stack_store(val, self.slot, self.offsets[field_index as usize]);
    }

    fn to_val(&self, builder: &mut FunctionBuilder, ty: Type) -> Val {
        if self.offsets.len() <= 1 {
            Val::Scalar(builder.ins().stack_load(ty, self.slot, 0))
        } else {
            let mut vals = Vec::new();
            for offset in &self.offsets {
                vals.push(Val::Scalar(
                    builder.ins().stack_load(ty, self.slot, *offset),
                ))
            }
            Val::Aggregate(vals)
        }
    }
}

#[test]
fn test_val_values() {
    let v0 = Val::Scalar(Value::from_u32(0));
    let v1 = Val::Scalar(Value::from_u32(1));
    let val = Val::Aggregate(vec![v0.clone(), v1.clone()]);
    assert_eq!(val.values().len(), 2);

    let v2 = Val::Scalar(Value::from_u32(2));
    let v3 = Val::Scalar(Value::from_u32(3));
    let val = Val::Aggregate(vec![
        v0,
        Val::Aggregate(vec![v1, Val::Aggregate(vec![v2]), v3]),
    ]);
    assert_eq!(val.values().len(), 4);
}

pub trait CraneliftModule: Module {
    fn finalize(self: Box<Self>, id: FuncId, output_file: &Path) -> i64;
}

impl CraneliftModule for JITModule {
    fn finalize(mut self: Box<Self>, id: FuncId, _output_file: &Path) -> i64 {
        self.finalize_definitions();
        let main = self.get_finalized_function(id);
        // let main: fn(i64, i64) -> i64 = unsafe { mem::transmute(main) };
        // let result = main(3, 4);
        let main: fn() -> i64 = unsafe { mem::transmute(main) };
        let result = main();
        // let result = code_fn(input, input);
        println!("JIT result: {}", result);
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

pub struct State {
    // scope: usize,
    // latest_scope: usize,
    var_index: u32,
    variables: HashMap<u32, Var>,
    pub module: Box<dyn CraneliftModule>,
    stack_vars: HashMap<u32, StackVar>,
}

impl State {
    ///
    pub fn compile_stmt(
        &mut self,
        input: &Input,
        b: &mut FunctionBuilder,
        node_id: NodeId,
        ty: Type,
    ) {
        let node = input.node(node_id);
        match node.tag {
            Tag::Assign => {
                // lhs: expr
                // rhs: expr
                println!("Compiling assign statment.");
                let val = self.compile_expr(input, b, node.rhs, ty).value();
                let lvalue = input.node(node.lhs);
                match lvalue.tag {
                    // a.x = b
                    Tag::Access => {
                        let mut identifier_id = lvalue.lhs;
                        while input.node(identifier_id).tag == Tag::Access {
                            identifier_id = input.node(identifier_id).lhs;
                        }
                        let field_index =
                            if let Definition::User(fi) = input.get_definition(lvalue.rhs) {
                                *fi
                            } else {
                                0
                            };
                        // Local variable
                        // let var = self.lookup_var(input, lhs.lhs);
                        // var.set_field(builder, val, field_index);
                        // Stack variable
                        let stack_var = self.lookup_stack_var(input, identifier_id);
                        stack_var.set_field(b, val, field_index);
                    }
                    // @a = b
                    Tag::Dereference => {
                        dbg!(lvalue.lhs);
                        let stack_var = self.lookup_stack_var(input, lvalue.lhs);
                        let ptr = stack_var.to_val(b, ty).value();
                        b.ins().store(MemFlags::new(), val, ptr, 0);
                    }
                    // a = b
                    Tag::Identifier => {
                        // Local variable
                        // let var = self.lookup_var(input, node.lhs);
                        // var.set_value(builder, val);
                        // Stack variable
                        let stack_var = self.lookup_stack_var(input, node.lhs);
                        stack_var.set_value(b, val);
                    }
                    _ => unreachable!("Invalid lvalue for assignment"),
                };
            }
            Tag::If => {
                let condition_expr = self.compile_expr(input, b, node.lhs, ty).value();
                let then_block = b.create_block();
                let merge_block = b.create_block();
                let body = input.node(node.rhs);
                b.ins().brz(condition_expr, merge_block, &[]);
                b.ins().jump(then_block, &[]);
                b.seal_block(then_block);
                // then block
                b.switch_to_block(then_block);
                for i in body.lhs..body.rhs {
                    let index = input.tree.node_index(i);
                    self.compile_stmt(input, b, index, ty);
                }
                if !b.is_filled() {
                    b.ins().jump(merge_block, &[]);
                }
                b.seal_block(merge_block);
                // merge block
                b.switch_to_block(merge_block);
            }
            Tag::IfElse => {
                let mut if_nodes = Vec::new();
                let mut then_blocks = Vec::new();
                for i in node.lhs..node.rhs {
                    let index = input.node_index(i);
                    let if_node = input.node(index);
                    if_nodes.push(if_node);
                    then_blocks.push(b.create_block());
                }
                let if_count = if if_nodes.last().unwrap().lhs == 0 {
                    if_nodes.len() - 1
                } else {
                    if_nodes.len()
                };
                let merge_block = b.create_block();
                for i in 0..if_count {
                    let condition_expr = self.compile_expr(input, b, if_nodes[i].lhs, ty).value();
                    b.ins().brnz(condition_expr, then_blocks[i], &[]);
                    b.seal_block(then_blocks[i]);
                    if i < if_count - 1 {
                        let block = b.create_block();
                        b.ins().jump(block, &[]);
                        b.seal_block(block);
                        b.switch_to_block(block);
                    }
                    if i == if_nodes.len() - 1 {
                        b.ins().jump(merge_block, &[]);
                    }
                }
                for i in if_count..if_nodes.len() {
                    b.ins().jump(then_blocks[i], &[]);
                    b.seal_block(then_blocks[i]);
                }
                for (i, if_node) in if_nodes.iter().enumerate() {
                    b.switch_to_block(then_blocks[i]);
                    let body = input.node(if_node.rhs);
                    for j in body.lhs..body.rhs {
                        let index = input.node_index(j);
                        self.compile_stmt(input, b, index, ty);
                    }
                    b.ins().jump(merge_block, &[]);
                }
                b.seal_block(merge_block);
                b.switch_to_block(merge_block);
            }
            Tag::VariableDecl => {
                // lhs: type
                // rhs: expr
                println!("Compiling variable declaration statement.");
                dbg!(input.node_type(node_id));

                // Local variables
                // let var = self.create_struct_var(input, index);
                // for variable in var.variables() {
                //     builder.declare_var(variable, ty);
                //     let val = if node.rhs != 0 {
                //         self.compile_expr(input, builder, node.rhs, ty)
                //             .value()
                //     } else {
                //         builder.ins().iconst(ty, 0)
                //     };
                //     builder.def_var(variable, val);
                // }

                // Stack variables
                let var = self.create_stack_var(input, b, node_id);
                if node.rhs != 0 {
                    // Assume the init_expr is a scalar
                    let value = self.compile_expr(input, b, node.rhs, ty).value();
                    b.ins().stack_store(value, var.slot, 0);
                }
                self.stack_vars.insert(node_id, var);
            }
            Tag::Return => {
                let mut return_values = Vec::new();
                for i in node.lhs..node.rhs {
                    let ni = input.node_index(i);
                    let val = self.compile_expr(input, b, ni, ty).value();
                    return_values.push(val);
                }
                b.ins().return_(&return_values[..]);
            }
            Tag::While => {
                let condition = self.compile_expr(input, b, node.lhs, ty).value();
                let while_block = b.create_block();
                let merge_block = b.create_block();
                // check condition
                // true? jump to loop body
                b.ins().brnz(condition, while_block, &[]);
                // false? jump to after loop
                b.ins().jump(merge_block, &[]);
                // block_while:
                b.switch_to_block(while_block);
                let body = input.node(node.rhs);
                for i in body.lhs..body.rhs {
                    let ni = input.node_index(i);
                    self.compile_stmt(input, b, ni, ty);
                }
                let condition = self.compile_expr(input, b, node.lhs, ty).value();
                // brnz block_while
                b.ins().brnz(condition, while_block, &[]);
                b.seal_block(while_block);
                b.ins().jump(merge_block, &[]);
                b.seal_block(merge_block);
                // block_merge:
                b.switch_to_block(merge_block);
            }
            _ => {
                self.compile_expr(input, b, node_id, ty);
            }
        }
    }

    ///
    pub fn compile_expr(
        &mut self,
        input: &Input,
        b: &mut FunctionBuilder,
        node_id: NodeId,
        ty: Type,
    ) -> Val {
        let node = input.node(node_id);
        dbg!(node.tag);
        match node.tag {
            Tag::Access => {
                let mut identifier_id = node.lhs;
                while input.node(identifier_id).tag == Tag::Access {
                    identifier_id = input.node(identifier_id).lhs;
                }
                // Get the struct stack variable.
                let field_index = if let Definition::User(fi) = input.get_definition(node.rhs) {
                    *fi
                } else {
                    0
                };

                // Local variables
                // let var = self.lookup_var(input, node.lhs);
                // let variables = var.variables();
                // let value = builder.use_var(variables[field_index as usize]);

                // Stack variables

                println!(
                    "Compiling access expr: {}.{}.",
                    input.tree.node_lexeme(identifier_id),
                    input.tree.node_lexeme(node.rhs)
                );
                dbg!(input.node_type(node_id));
                let offsets = self.node_type_to_offsets(input, node_id);
                dbg!(&offsets);
                if offsets.len() <= 1 {
                    let stack_var = self.lookup_stack_var(input, identifier_id);
                    let value = b.ins().stack_load(
                        ty,
                        stack_var.slot,
                        stack_var.offsets[field_index as usize],
                    );
                    Val::Scalar(value)
                } else {
                    let mut vals = Vec::new();
                    for offset in &offsets {
                        let stack_var = self.lookup_stack_var(input, identifier_id);
                        let value = b.ins().stack_load(
                            ty,
                            stack_var.slot,
                            stack_var.offsets[field_index as usize] + offset,
                        );
                        vals.push(Val::Scalar(value));
                    }
                    Val::Aggregate(vals)
                }
            }
            Tag::Address => {
                let var = self.lookup_stack_var(input, node.lhs);
                let val = b.ins().stack_addr(ty, var.slot, 0);
                Val::Scalar(val)
            }
            Tag::Dereference => {
                let var = self.lookup_stack_var(input, node.lhs);
                let val = var.to_val(b, ty).value();
                let val = b.ins().load(ty, MemFlags::new(), val, 0);
                Val::Scalar(val)
            }
            Tag::Add => {
                let (lhs, rhs) = self.compile_children(input, b, node, ty);
                Val::Scalar(b.ins().iadd(lhs, rhs))
            }
            Tag::Sub => {
                let (lhs, rhs) = self.compile_children(input, b, node, ty);
                Val::Scalar(b.ins().isub(lhs, rhs))
            }
            Tag::Div => {
                let (lhs, rhs) = self.compile_children(input, b, node, ty);
                Val::Scalar(b.ins().sdiv(lhs, rhs))
            }
            Tag::Mul => {
                let (lhs, rhs) = self.compile_children(input, b, node, ty);
                Val::Scalar(b.ins().imul(lhs, rhs))
            }
            Tag::Greater => {
                let (lhs, rhs) = self.compile_children(input, b, node, ty);
                Val::Scalar(b.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs))
            }
            Tag::Less => {
                let (lhs, rhs) = self.compile_children(input, b, node, ty);
                Val::Scalar(b.ins().icmp(IntCC::SignedLessThan, lhs, rhs))
            }
            Tag::Grouping => self.compile_expr(input, b, node.lhs, ty),
            Tag::IntegerLiteral => {
                let token_str = input.node_lexeme_offset(node, 0);
                let value = token_str.parse::<i64>().unwrap();
                Val::Scalar(b.ins().iconst(ty, value))
            }
            Tag::Call => {
                let mut sig = self.module.make_signature();
                let function = input.node(node.lhs);
                let name = input.node_lexeme_offset(function, 0);

                // Arguments
                let arguments = input.node(node.rhs);
                let mut args = Vec::new();
                for i in arguments.lhs..arguments.rhs {
                    let index = input.node_index(i);
                    let mut values = self.compile_expr(input, b, index, ty).values();
                    for _ in &values {
                        sig.params.push(AbiParam::new(ty));
                    }
                    args.append(&mut values);
                }

                // Assume one return value.
                let return_type = input.type_id(node_id);
                if return_type != TypeIndex::Void as TypeId {
                sig.returns.push(AbiParam::new(ty));
                }

                let callee = self
                    .module
                    .declare_function(name, Linkage::Import, &sig)
                    .unwrap();
                let local_callee = self.module.declare_func_in_func(callee, b.func);
                let call = b.ins().call(local_callee, &args);
                if return_type != TypeIndex::Void as TypeId {
                Val::Scalar(b.inst_results(call)[0])
                } else {
                    Val::Scalar(b.ins().iconst(ty, 0))
                }
            }
            Tag::Identifier => {
                // Local variable
                // let var = self.lookup_var(input, index);
                // let val = var.to_val(b);

                // Stack variable
                let var = self.lookup_stack_var(input, node_id);
                let val = var.to_val(b, ty);

                val
            }
            _ => Val::Scalar(b.ins().iconst(ty, 0)),
        }
    }

    fn compile_children(
        &mut self,
        input: &Input,
        b: &mut FunctionBuilder,
        node: &Node,
        ty: Type,
    ) -> (Value, Value) {
        let lhs = self.compile_expr(input, b, node.lhs, ty).value();
        let rhs = self.compile_expr(input, b, node.rhs, ty).value();
        (lhs, rhs)
    }

    /// Maps node_id -> Var.
    fn lookup_var(&self, input: &Input, node_id: u32) -> Var {
        println!("lookup_var: \"{}\"", node_id);
        if let Definition::User(decl_id) = input.get_definition(node_id) {
            println!("decl_id: \"{}\"", decl_id);
            let var = self
                .variables
                .get(&decl_id)
                .expect("failed to get variable")
                .clone();
            var
        } else {
            panic!();
        }
    }

    /// Maps node_id -> StackVar.
    fn lookup_stack_var(&self, input: &Input, node_id: u32) -> StackVar {
        if let Definition::User(decl_id) = input.get_definition(node_id) {
            let var = self
                .stack_vars
                .get(&decl_id)
                .expect("failed to get variable")
                .clone();
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

    /// Creates a Var for the given node's type and inserts it into the variable map.
    fn create_struct_var(&mut self, input: &Input, node_id: u32) -> Var {
        let var = self.node_type_to_var(input, node_id);
        self.variables.insert(node_id, var.clone());
        var
    }

    fn create_stack_var(
        &mut self,
        input: &Input,
        b: &mut FunctionBuilder,
        node_id: u32,
    ) -> StackVar {
        let size = input.sizeof(input.type_id(node_id));
        dbg!(size);
        let slot = b.create_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, size));
        let offsets = self.node_type_to_offsets(input, node_id);
        StackVar { slot, offsets }
    }

    fn node_type_to_var(&mut self, input: &Input, node_id: NodeId) -> Var {
        let type_id = input.node_types[node_id as usize];
        self.type_to_var(input, type_id)
    }

    fn node_type_to_offsets(&mut self, input: &Input, node_id: NodeId) -> Vec<i32> {
        let type_id = input.node_types[node_id as usize];
        let mut offsets = Vec::new();
        self.type_to_offsets(input, type_id, &mut offsets);
        offsets
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

    fn type_to_offsets(&mut self, input: &Input, type_id: usize, offsets: &mut Vec<i32>) {
        let t = &input.types[type_id];
        // dbg!(t);
        match t {
            Typ::Struct { fields } => {
                for type_id in fields {
                    self.type_to_offsets(input, *type_id, offsets);
                }
            }
            _ => {
                let offset = if let Some(last) = offsets.last() {
                    last + 8
                } else {
                    0
                };
                offsets.push(offset);
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
    pub definitions: &'a HashMap<u32, Definition>,
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

    fn get_definition(&self, node_id: NodeId) -> &Definition {
        self.definitions.get(&node_id).unwrap()
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

    pub fn type_id(&self, node_id: NodeId) -> usize {
        self.node_types[node_id as usize]
    }

    pub fn node_type(&self, node_id: NodeId) -> &Typ {
        &self.types[self.node_types[node_id as usize]]
    }
}

pub struct Generator<'a> {
    pub builder_ctx: FunctionBuilderContext,
    pub ctx: codegen::Context,
    data_ctx: DataContext,
    input: &'a Input<'a>,
    pub state: State,
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
            let mut jit_builder =
                JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
            let print_int_addr = print_internal as *const u8;
            jit_builder.symbol("print_int", print_int_addr);
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
            input,
            state: State {
                module,
                var_index: 0,
                variables: HashMap::new(),
                stack_vars: HashMap::new(),
            },
        }
    }

    ///
    pub fn init_signature<'b>(
        func: &mut codegen::ir::Function,
        input: &Input,
        parameters: &Node,
        returns: &Node,
        t: Type,
    ) {
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
    }

    fn push_params(input: &Input, node_id: NodeId, params: &mut Vec<AbiParam>, t: Type) {
        let mut stack = vec![node_id];
        loop {
            if let Some(node_id) = stack.pop() {
                let field_node = input.node(node_id);
                let type_node_id = field_node.rhs;
                if let Definition::User(def_id) = input.get_definition(type_node_id) {
                    let def_node = input.node(*def_id);
                    if def_node.tag == Tag::Struct {
                        for i in def_node.lhs..def_node.rhs {
                            let field_id = input.node_index(i);
                            stack.push(field_id);
                        }
                        continue;
                    }
                }
                params.push(AbiParam::new(t))
            } else {
                return;
            }
        }
    }

    ///
    pub fn compile_nodes(mut self, filename: &Path) -> Option<i64> {
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
            return Some(self.state.module.finalize(id, filename));
        }
        println!("Finalized {:?}", func_ids);
        None
    }

    ///
    pub fn compile_function_decl(&mut self, index: u32) -> FuncId {
        dbg!("compile_function_decl");
        let int = self.state.module.target_config().pointer_type();
        let node = self.input.node(index);
        println!("{}", self.input.node_lexeme_offset(node, -1));
        assert_eq!(node.tag, Tag::FunctionDecl);
        let prototype = self.input.node(node.lhs);
        assert_eq!(prototype.tag, Tag::Prototype);
        let parameters = self.input.node(prototype.lhs);
        assert_eq!(parameters.tag, Tag::Parameters);
        let returns = self.input.node(prototype.rhs);
        Self::init_signature(&mut self.ctx.func, self.input, parameters, returns, int);
        let mut b = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);
        let entry_block = b.create_block();
        b.append_block_params_for_function_params(entry_block);
        b.switch_to_block(entry_block);
        b.seal_block(entry_block);

        dbg!(b.block_params(entry_block).len());

        // Define parameters as stack variables.
        let mut scalar_count = 0;
        for i in parameters.lhs..parameters.rhs {
            let ni = self.input.node_index(i);
            let var = self.state.create_stack_var(self.input, &mut b, ni);
            for &offset in &var.offsets {
                let value = b.block_params(entry_block)[scalar_count];
                b.ins().stack_store(value, var.slot, offset);
                scalar_count += 1;
            }
            self.state.stack_vars.insert(ni, var);
        }

        // Define parameters as local variables.
        // let mut scalar_count = 0;
        // for i in parameters.lhs..parameters.rhs {
        //     let ni = self.input.node_index(i);
        //     let var = self.state.create_struct_var(self.input, ni);
        //     for &variable in &var.variables() {
        //         builder.declare_var(variable, int);
        //         let val = builder.block_params(entry_block)[scalar_count];
        //         builder.def_var(variable, val);
        //         scalar_count += 1;
        //     }
        // }

        let body = self.input.node(node.rhs);
        for i in body.lhs..body.rhs {
            let index = self.input.node_index(i);
            self.state.compile_stmt(&self.input, &mut b, index, int);
        }

        b.finalize();
        let name = self.input.tree.node_lexeme_offset(node, -1);
        println!("{} {}", name, b.func.display());
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
}

extern "C" fn print_internal(value: isize) -> isize {
    println!("{}", value);
    0
}
