use crate::{
    analyze::{BuiltInFunction, Definition},
    parse::{Node, NodeId, Tag, Tree},
    types::{float_type, integer_type, Type, TypeId, TypeRef, T},
    workspace::{Result, Workspace},
};
use codespan_reporting::diagnostic::Diagnostic;
use std::collections::HashMap;

/**
 * 1. Infer the type of each expression from the types of its components.
 * 2. Check that the types of expressions in certain contexts match what is expected.
 *     - Recursively walk the AST.
 *     - For each statement:
 *        - Typecheck any subexpressions it contains.
 *           - Report errors if no type can be assigned.
 *           - Report errors if the wrong type is assigned.
 *     - Typecheck child statements.
 *     - Check the overall correctness.
 */

type TypeMap<P = Vec<TypeId>> = HashMap<P, TypeId>;
type TypeParameters = HashMap<NodeId, HashMap<Vec<TypeId>, Vec<TypeId>>>;

pub struct Typechecker<'a> {
    workspace: &'a mut Workspace,
    tree: &'a mut Tree,
    definitions: &'a mut HashMap<NodeId, Definition>,
    overload_sets: &'a HashMap<NodeId, Vec<Definition>>,

    array_types: TypeMap<(TypeId, usize)>,
    builtin_function_types: TypeMap<BuiltInFunction>,
    function_types: TypeMap<Type>,
    pointer_types: TypeMap<TypeId>,
    polymorphic_types: HashMap<NodeId, TypeMap>,
    tuple_types: TypeMap,

    /// Maps TypeId to declaration NodeId.
    type_definitions: HashMap<TypeId, NodeId>,
    type_parameters: TypeParameters,
    current_parameters: Vec<TypeId>,
    // This will have to be updated for nested struct definitions.
    current_fn_type_id: Option<TypeId>,
    current_struct_id: NodeId,

    fn_id_stack: Vec<NodeId>,

    next_variable_id: usize,

    pub types: Vec<Type>,
}

impl<'a> Typechecker<'a> {
    pub fn new(
        workspace: &'a mut Workspace,
        tree: &'a mut Tree,
        definitions: &'a mut HashMap<u32, Definition>,
        overload_sets: &'a HashMap<NodeId, Vec<Definition>>,
    ) -> Self {
        let mut types = vec![Type::None; T::Count as usize];
        // types[T::None as TypeId] = Type::None;
        types[T::Void as TypeId] = Type::Void;
        types[T::Never as TypeId] = Type::Never;
        types[T::Any as TypeId] = Type::Any;
        types[T::Boolean as TypeId] = Type::Boolean;
        // Signed integers
        types[T::I8 as TypeId] = integer_type(false, true, 1);
        types[T::I16 as TypeId] = integer_type(false, true, 2);
        types[T::I32 as TypeId] = integer_type(false, true, 4);
        types[T::I64 as TypeId] = integer_type(false, true, 8);

        types[T::CI8 as TypeId] = integer_type(true, true, 1);
        types[T::CI16 as TypeId] = integer_type(true, true, 2);
        types[T::CI32 as TypeId] = integer_type(true, true, 4);
        types[T::CI64 as TypeId] = integer_type(true, true, 8);

        // Unsigned integers
        types[T::U8 as TypeId] = integer_type(false, false, 1);
        types[T::U16 as TypeId] = integer_type(false, false, 2);
        types[T::U32 as TypeId] = integer_type(false, false, 4);
        types[T::U64 as TypeId] = integer_type(false, false, 8);
        // Floating-point numbers
        types[T::F32 as TypeId] = float_type(false, 4);
        types[T::F64 as TypeId] = float_type(false, 8);
        types[T::Type as TypeId] = Type::Type;

        // Set up string type.
        // types.push(Type::Pointer {
        //     typ: T::U8 as TypeId,
        //     is_generic: false,
        // });
        // let ptr_u8_type = types.len() - 1;

        let pointer_types = HashMap::new();
        // let pointer_types = HashMap::from([(T::U8 as TypeId, ptr_u8_type)]);

        // types[T::String as TypeId] = Type::Struct {
        //     fields: vec![ptr_u8_type, T::Integer as TypeId],
        //     is_generic: false,
        // };

        // let mut types = vec![
        //     Type::Void,
        //     Type::Boolean,
        //     Type::Integer,
        //     Type::Float,
        //     Type::String,
        //     Type::Type,
        // ];

        let mut builtin_function_types = HashMap::new();
        // Add
        let binary_int_op_type = Type::Function {
            parameters: vec![T::I64 as TypeId, T::I64 as TypeId],
            returns: vec![T::I64 as TypeId],
        };
        types.push(binary_int_op_type);
        let binary_int_op_type_id = types.len() - 1;
        // Add i8
        let add_i8_type = types.len();
        types.push(Type::Function {
            parameters: vec![T::I8 as TypeId, T::I8 as TypeId],
            returns: vec![T::I8 as TypeId],
        });
        // sizeof
        types.push(Type::Function {
            parameters: vec![T::Any as TypeId],
            returns: vec![T::I64 as TypeId],
        });

        let sizeof_type_id = types.len() - 1;
        let fn_types = [
            (BuiltInFunction::Add, binary_int_op_type_id),
            (BuiltInFunction::AddI8, add_i8_type),
            (BuiltInFunction::Mul, binary_int_op_type_id),
            (BuiltInFunction::SizeOf, sizeof_type_id),
            (BuiltInFunction::SubI64, binary_int_op_type_id),
        ];
        for (tag, type_id) in fn_types {
            builtin_function_types.insert(tag, type_id);
        }

        Self {
            workspace,
            tree,
            definitions,
            overload_sets,
            array_types: HashMap::new(),
            builtin_function_types,
            function_types: HashMap::new(),
            pointer_types,
            polymorphic_types: HashMap::new(),
            tuple_types: HashMap::new(),
            type_definitions: HashMap::new(),
            type_parameters: HashMap::new(),
            current_parameters: vec![],
            current_fn_type_id: None,
            current_struct_id: 0,
            fn_id_stack: vec![],
            next_variable_id: 0,
            types,
        }
    }

    pub fn results(self) -> (Vec<Type>, TypeParameters) {
        (self.types, self.type_parameters)
    }

    pub fn typecheck(&mut self) {
        println!("{}", crate::format_red!("---- DETECT ----"));
        if let Err(diagnostic) = self.detect(0) {
            self.workspace.diagnostics.push(diagnostic);
            return;
        }

        println!("{}", crate::format_red!("---- INFER ----"));
        if let Err(diagnostic) = self.infer(0) {
            self.workspace.diagnostics.push(diagnostic);
            return;
        }

        self.print();
        println!("{:#?}", self.tree);

        println!("{}", crate::format_red!("---- CHECK ----"));
        if let Err(diagnostic) = self.check(0) {
            self.workspace.diagnostics.push(diagnostic);
            return;
        }

        self.print();
        println!("{:#?}", self.tree);

        println!("{}", crate::format_red!("---- FINALIZE ----"));
        if let Err(diagnostic) = self.finalize(0) {
            self.workspace.diagnostics.push(diagnostic);
            return;
        }

        for ni in 1..self.tree.nodes.len() {
            let t = self.get(ni as NodeId);
            if let Type::Parameter { binding, .. } = &self.types[t] {
                self.set(ni as NodeId, *binding);
            }
        }
    }

    // Detects nodes that represent types.
    fn detect(&mut self, node_id: NodeId) -> Result<()> {
        let node = &self.tree.node(node_id).clone();
        match node.tag {
            Tag::Root | Tag::Module => {
                for i in self.tree.range(node) {
                    self.detect(self.tree.node_index(i))?;
                }
            }
            // Declarations
            Tag::FunctionDecl => {
                let type_id = self.infer(node.lhs).unwrap();
                self.type_definitions.insert(type_id, node_id);
                // self.detect(node.rhs);
            }
            Tag::Struct => {
                // let type_id = self.infer(node_id).unwrap();
                // self.type_definitions.insert(type_id, node_id);
                // if node.lhs != 0 {
                //     let type_parameters = self.tree.node(node.lhs);
                //     let lhs = type_parameters.lhs;
                //     for i in lhs..type_parameters.rhs {
                //         let node_id = self.tree.node_index(i);
                //         let param_type = self.add_type(Type::Parameter {
                //             index: (i - lhs) as usize,
                //             binding: T::Void as TypeId,
                //         });
                //         self.set_t(node_id, param_type);
                //     }
                //     self.polymorphic_types
                //         .try_insert(node_id, HashMap::new())
                //         .expect("tried to insert duplicate polymorphic type map");
                // };
                // // Infer field types
                // let mut fields = Vec::new();
                // for i in self.tree.range(node) {
                //     self.infer_node(self.tree.node_index(i))?;
                // }
                // // Set struct type based on field types
                // for i in self.tree.range(node) {
                //     let ni = self.tree.node_index(i);
                //     fields.push(self.get_t(ni));
                // }
                // let is_generic = fields.iter().any(|&type_id| self.is_generic(type_id));
                // let struct_type = Type::Struct { fields, is_generic };
                // let type_id = if self.tree.name(node_id) == "String" {
                //     self.types[T::String as TypeId] = struct_type;
                //     T::String as TypeId
                // } else {
                //     self.add_type(struct_type)
                // };
                // self.type_definitions.insert(type_id, node_id);
                // type_id
            }
            Tag::VariableDecl => {
                // let annotation_id = self.tree.node_extra(node, 0);
                // let type_id = if annotation_id != 0 {
                //     self.detect(annotation_id);
                //     self.get_node_type(node_id)
                // } else {
                //     self.new_unification_variable()
                // };

                // let rvalues_id = self.tree.node_extra(node, 1);
                // if rvalues_id != 0 {
                //     self.detect(rvalues_id);
                // }
                // self.set_node_type(node_id, type_id);
            }
            // Expressions
            Tag::Identifier => {}
            // Literal values
            // Tag::Return => {}
            Tag::Import => {}
            _ => {
                panic!("{:?}", node.tag);
            }
        }
        Ok(())
    }

    /// Computes a type for every node.
    /// If a type can't be inferred yet, create a type variable.
    fn infer(&mut self, node_id: NodeId) -> Result<TypeId> {
        let node = &self.tree.node(node_id).clone();
        // println!(
        //     "infer {:>4}. {:<12} {:?}",
        //     node_id,
        //     self.tree.name(node_id),
        //     node.tag,
        // );
        match node.tag {
            Tag::Root | Tag::Module | Tag::Parameters | Tag::Block | Tag::IfElse => {
                for i in self.tree.range(node) {
                    let ni = self.tree.node_index(i);
                    self.infer(ni)?;
                }
            }
            Tag::Import => {}
            Tag::Expressions => {
                let mut types = vec![];
                for i in self.tree.range(node) {
                    types.push(self.infer(self.tree.node_index(i))?);
                }
                let type_id = self.add_tuple_type(types);
                self.set(node_id, type_id);
            }
            // Declarations
            Tag::FunctionDecl => {
                self.infer(node.lhs)?;
                let fn_type_id = self.get(node.lhs);
                // Immediately set the declaration's type to handle recursion.
                self.set(node_id, fn_type_id);
                // Body
                if node.rhs != 0 {
                    if self.current_fn_type_id.is_none() {
                        self.current_fn_type_id = Some(fn_type_id);
                    }
                    self.fn_id_stack.push(node_id);
                    self.infer(node.rhs)?;
                    self.fn_id_stack.pop();
                    self.current_fn_type_id = None;
                }
            }
            Tag::Prototype => {
                let mut parameters = Vec::new();
                let mut returns = Vec::new();

                let type_parameters_id = node.lhs;
                if type_parameters_id != 0 {
                    // Type parameters
                    let type_parameters = self.tree.node(type_parameters_id);
                    let start = type_parameters.lhs;
                    let end = type_parameters.rhs;
                    for i in start..end {
                        let node_id = self.tree.node_index(i);
                        let param_type = self.add_type(Type::Parameter {
                            index: (i - start) as usize,
                            binding: T::Void as TypeId,
                        });
                        self.set(node_id, param_type);
                    }
                }

                let parameters_id = self.tree.node_extra(node, 0);
                self.infer(parameters_id)?; // parameters

                let returns_id = self.tree.node_extra(node, 1);
                let return_type = if returns_id != 0 {
                    self.infer(returns_id)? // returns
                } else {
                    T::Void as TypeId
                };

                let params = self.tree.node(parameters_id);
                let rets = self.tree.node(returns_id);

                assert_eq!(params.tag, Tag::Parameters);
                for i in params.lhs..params.rhs {
                    let ni = self.tree.node_index(i);
                    parameters.push(self.get(ni));
                }
                if rets.tag == Tag::Expressions {
                    for i in rets.lhs..rets.rhs {
                        let ni = self.tree.node_index(i);
                        returns.push(self.get(ni));
                    }
                } else if rets.tag == Tag::Identifier
                    || rets.tag == Tag::Type && return_type != T::Void as TypeId
                {
                    returns.push(return_type);
                }
                let fn_type = self.add_function_type(Type::Function {
                    parameters,
                    returns,
                });
                self.set(node_id, fn_type);
            }
            Tag::Struct => {
                if node.lhs != 0 {
                    let type_parameters = self.tree.node(node.lhs);
                    let lhs = type_parameters.lhs;
                    for i in lhs..type_parameters.rhs {
                        let node_id = self.tree.node_index(i);
                        let param_type = self.add_type(Type::Parameter {
                            index: (i - lhs) as usize,
                            binding: T::Void as TypeId,
                        });
                        self.set(node_id, param_type);
                    }
                    self.polymorphic_types
                        .try_insert(node_id, HashMap::new())
                        .expect("tried to insert duplicate polymorphic type map");
                };
                // Infer field types
                let mut fields = Vec::new();
                for i in self.tree.range(node) {
                    self.infer(self.tree.node_index(i))?;
                }
                // Set struct type based on field types
                for i in self.tree.range(node) {
                    let ni = self.tree.node_index(i);
                    fields.push(self.get(ni));
                }
                let is_generic = fields.iter().any(|&type_id| self.is_generic(type_id));
                let struct_type = Type::Struct { fields, is_generic };
                let type_id = if self.tree.name(node_id) == "String" {
                    self.types[T::String as TypeId] = struct_type;
                    T::String as TypeId
                } else {
                    self.add_type(struct_type)
                };
                self.type_definitions.insert(type_id, node_id);
                self.set(node_id, type_id);
            }
            Tag::Field => {
                let annotation_id = self.tree.node_extra(node, 0);
                let type_id = self.infer(annotation_id)?;
                self.set(node_id, type_id);
            }
            Tag::VariableDecl => {
                // Detect
                let identifiers = &self.tree.node(node.lhs).clone();
                let annotation_id = self.tree.node_extra(node, 0);
                let rvalues_id = self.tree.node_extra(node, 1);

                // panic!("{}", annotation);

                // Infer
                if rvalues_id == 0 {
                    let annotation_type = self.infer(annotation_id)?;
                    // Set lhs types
                    for i in self.tree.range(identifiers) {
                        let ni = self.tree.node_index(i);
                        self.set(ni, annotation_type);
                    }
                } else {
                    // Infer expression types
                    self.infer(rvalues_id)?;

                    // Unify lhs and rhs types
                    let mut rtypes = Vec::<TypeId>::new();
                    let rvalues = self.tree.node(rvalues_id);
                    for i in self.tree.range(rvalues) {
                        let ni = self.tree.node_index(i);
                        let ti = self.get(ni);
                        match &self.types[ti] {
                            Type::Tuple { fields } => rtypes.extend(fields),
                            _ => rtypes.push(ti),
                        }
                    }

                    let annotation_type = if annotation_id == 0
                        && rtypes.iter().all(|rtype| is_integer_literal(*rtype))
                    {
                        T::I64 as TypeId
                    } else {
                        self.new_type_variable()
                    };

                    // Set lhs types
                    for i in self.tree.range(identifiers) {
                        let ni = self.tree.node_index(i);
                        self.set(ni, annotation_type);
                    }

                    dbg!(&rtypes);

                    for (i, rtype) in self.tree.range(identifiers).zip(rtypes.iter()) {
                        let ni = self.tree.node_index(i);
                        let ltype = self.get(ni);
                        dbg!(&self.types[*rtype]);
                        dbg!(&self.types[ltype]);
                        if let Some(unified_type) = self.unify(*rtype, ltype) {
                            // dbg!(&self.types[*ti]);
                            dbg!(&self.types[*rtype]);
                            dbg!(&self.types[ltype]);
                            dbg!(&self.types[unified_type]);
                            dbg!(&unified_type);

                            // Fixes subtraction, breaks boolean.hb
                            // self.set_node_type(ni, unified_type);
                        } else {
                            panic!()
                        }
                    }

                    // Set tuple type
                    // let tuple_type = self.add_tuple_type(rtypes);
                    // self.set_node_type(rvalues_id, tuple_type);
                }
            }
            // Statements
            Tag::If | Tag::While => {
                if node.lhs != 0 {
                    self.infer(node.lhs)?;
                }
                self.infer(node.rhs)?;
            }
            Tag::Assign => {
                let a = self.infer(node.lhs)?;
                let b = self.infer(node.rhs)?;

                // println!("a: {a} = b: {b}");
                // dbg!(self.unify(b, a));
                // self.set_node_type(node.rhs, self.get_node_type(node.lhs));
                // println!("a: {a} = b: {b}");
                // dbg!(self.get_t(node.rhs));

                self.is_subtype(b, a);
            }
            Tag::Return => {
                // Infer children
                let mut types = vec![];
                for i in self.tree.range(node) {
                    let ti = self.infer(self.tree.node_index(i))?;
                    types.push(ti);
                }
                dbg!(&types);
                let expr_type = match types.len() {
                    0 => T::Void as TypeId,
                    1 => types[0],
                    _ => self.add_tuple_type(types),
                };
                if expr_type != 0 {
                    let node_id = self.fn_id_stack.last().unwrap();
                    dbg!(node_id);
                    let fn_type = self.get(*node_id);
                    // let var = self.new_unification_variable();
                    dbg!(&self.types[fn_type]);

                    let fn_return_type = self.add_tuple_type(self.types[fn_type].returns().clone());
                    // Unify function type with that of function returning expr_type

                    // Unify return type with that of function return

                    self.unify(expr_type, fn_return_type);
                    // self.print();
                    // println!(
                    //     "{} vs {} -> {}",
                    //     expr_type,
                    //     self.types[fn_type].returns()[0],
                    //     unified_type.unwrap(),
                    // );
                }
            }
            // Operator expressions
            Tag::Access => {
                self.infer(node.lhs)?;
                let var = self.new_type_variable();
                self.set(node_id, var);
                // Cannot typecheck rhs until lhs is inferred.
            }
            Tag::Address => {
                self.infer(node.lhs)?;
                let var = self.new_type_variable();
                self.unify(var, self.get(node.lhs));
                if !is_none(self.get(node.lhs)) {
                    let ptr = self.add_pointer_type(var);
                    self.set(node_id, ptr);
                } else {
                    let var = self.new_type_variable();
                    let ptr = self.add_pointer_type(var);
                    self.set(node_id, ptr)
                }
            }
            Tag::Call => {
                // Argument expressions
                self.infer(node.rhs)?;
                let var = self.new_type_variable();
                self.set(node_id, var);
            }
            Tag::Add | Tag::Mul => {
                self.infer(node.lhs)?;
                self.infer(node.rhs)?;
                let var = self.new_type_variable();
                self.set(node_id, var);
            }
            Tag::Div
            | Tag::Sub
            | Tag::BitwiseAnd
            | Tag::BitwiseOr
            | Tag::BitwiseShiftL
            | Tag::BitwiseShiftR
            | Tag::BitwiseXor => {
                let ltype = self.infer(node.lhs)?;
                let rtype = self.infer(node.rhs)?;
                let t = if let Some(utype) = self.unify(ltype, rtype) {
                    utype
                } else {
                    self.new_type_variable()
                };
                // let t = self.new_unification_variable();
                // dbg!(&self.types[t]);
                self.set(node_id, t);
                // dbg!(&self.types[self.get_ty(node_id)]);
            }
            Tag::Negation => {
                let ltype = self.infer(node.lhs)?;
                self.set(node_id, ltype);
            }
            Tag::Equality
            | Tag::Greater
            | Tag::GreaterEqual
            | Tag::Inequality
            | Tag::Less
            | Tag::LessEqual
            | Tag::LogicalAnd
            | Tag::LogicalOr => {
                let ltype = self.infer(node.lhs)?;
                let rtype = self.infer(node.rhs)?;
                // self.unify(ltype, rtype);
                if ltype != rtype {
                    if is_integer(ltype) && is_integer_literal(rtype) {
                        self.set(node.rhs, ltype);
                    } else if is_integer_literal(ltype) && is_integer(rtype) {
                        self.set(node.lhs, rtype);
                    } else {
                        let utype = dbg!(self.unify(ltype, rtype)).unwrap();
                        self.set(node.lhs, utype);
                        self.set(node.rhs, utype);
                        // return Err(Diagnostic::error()
                        //     .with_message(format!(
                        //         "mismatched types: left is \"{:?}\", right is \"{:?}\"",
                        //         self.types[ltype], self.types[rtype]
                        //     ))
                        //     .with_labels(vec![self.tree.label(node.token)]));
                    }
                } else if is_integer_literal(ltype) && is_integer_literal(rtype) {
                    self.set(node.lhs, T::I64 as TypeId);
                    self.set(node.rhs, T::I64 as TypeId);
                }
                self.set(node_id, T::Boolean as TypeId);
            }
            Tag::Not => {
                self.infer(node.lhs)?;
                self.set(node_id, T::Boolean as TypeId);
            }
            Tag::Dereference => {
                self.infer(node.lhs)?; // This should be a pointer.
                let var = self.new_type_variable();
                let ptr = self.add_pointer_type(var);
                self.unify(ptr, self.get(node.lhs));
                self.set(node_id, var);
            }
            Tag::Subscript => {
                self.infer(node.lhs)?;
                self.infer(node.rhs)?;
                let var = self.new_type_variable();
                self.set(node_id, var);
            }
            // Simple expressions
            Tag::Identifier => {
                // The type of an identifier is the type of its definition.
                let decltype = self.get_identifier_type(node_id);
                self.set(node_id, decltype);
            }
            Tag::Type => {
                let definition = self.definitions.get(&node_id).unwrap();
                match definition {
                    Definition::BuiltIn(builtin) => match builtin {
                        T::Array => {
                            // Map concrete type arguments to an array type.
                            // Expect two type parameters.
                            assert_eq!(node.tag, Tag::Type);
                            if node.rhs - node.lhs != 2 {
                                // return Err(Diagnostic::error()
                                //     .with_message(format!(
                                //         "Expected 2 type parameters, got {}.",
                                //         node.rhs - node.lhs
                                //     ))
                                //     .with_labels(vec![self.tree.label(node.token)]));
                            }
                            let ni = self.tree.node_index(node.lhs);
                            let value_type = self.infer(ni)?;
                            let ni = self.tree.node_index(node.lhs + 1);
                            let length_node = self.tree.node(ni);
                            if length_node.tag != Tag::IntegerLiteral {
                                // return Err(Diagnostic::error()
                                //     .with_message(
                                //         "The length of an Array must be an integer literal.",
                                //     )
                                //     .with_labels(vec![self.tree.label(length_node.token)]));
                            }
                            let token_str = self.tree.node_lexeme(ni);
                            let length = token_str.parse::<i64>().unwrap();
                            let type_id = self.add_array_type(value_type, length as usize);
                            self.set(node_id, type_id)
                        }
                        T::Pointer => {
                            // Map concrete type argument to a pointer type.
                            // Expect one type parameter
                            if node.rhs - node.lhs != 1 {
                                // return Err(Diagnostic::error()
                                //     .with_message(format!(
                                //         "Expected 1 type parameter, got {}.",
                                //         node.rhs - node.lhs
                                //     ))
                                //     .with_labels(vec![self.tree.label(node.token)]));
                            }
                            let ni = self.tree.node_index(node.lhs);
                            let value_type = self.infer(ni)?;
                            let type_id = self.add_pointer_type(value_type);
                            self.set(node_id, type_id);
                        }
                        _ => {
                            // return Err(Diagnostic::error()
                            //     .with_message("Undefined built-in type")
                            //     .with_labels(vec![self.tree.label(node.token)]))
                        }
                    },
                    Definition::User(id) => {
                        // Map concrete type arguments to a struct type.
                        let struct_decl_id = *id;
                        self.current_struct_id = *id;
                        let current_parameter_count = self.current_parameters.len();
                        for i in self.tree.range(node) {
                            let ni = self.tree.node_index(i);
                            let param_type = self.infer(ni)?;
                            self.current_parameters.push(param_type);
                        }
                        let struct_type_id = self.get(struct_decl_id);
                        let specified_type = self.monomorphize_type(struct_type_id);
                        self.current_parameters.truncate(current_parameter_count);
                        self.current_struct_id = 0;
                        self.set(node_id, specified_type);
                    }
                    _ => {
                        // return Err(Diagnostic::error().with_message("Undefined type"))
                    }
                }
            }
            Tag::StringLiteral => self.set(node_id, T::String as TypeId),
            Tag::IntegerLiteral => {
                let token_str = self.tree.node_lexeme(node_id);
                self.set(
                    node_id,
                    smallest_integer_type(token_str.parse::<i64>().unwrap()),
                )
                // self.set_node_type(node_id, T::IntegerLiteral as TypeId)
            }
            Tag::True | Tag::False => self.set(node_id, T::Boolean as TypeId),
            Tag::FloatLiteral => self.set(node_id, T::F32 as TypeId),
            _ => {
                panic!("infer: {:?}", node.tag);
                // if node.lhs != 0 {
                //     self.infer(node.lhs)?;
                // }
                // if node.rhs != 0 {
                //     self.infer(node.rhs)?;
                // }
            }
        }
        // println!(
        //     "infer {:>4}. {:<12} {:?}: {}",
        //     node_id,
        //     self.tree.name(node_id),
        //     node.tag,
        //     self.get_t(node_id)
        // );
        Ok(self.get(node_id))
    }

    /// Verifies that all inferred types are used correctly.
    /// Types that are used correctly are made concrete and implicitly converted.
    fn check(&mut self, node_id: NodeId) -> Result<()> {
        let node = &self.tree.node(node_id).clone();
        println!("check: {node_id} {:?}", node.tag);
        match node.tag {
            Tag::Root | Tag::Module | Tag::Expressions | Tag::Block | Tag::IfElse => {
                for i in self.tree.range(node) {
                    self.check(self.tree.node_index(i))?;
                }
            }
            // Declarations
            Tag::FunctionDecl => {
                self.check(node.lhs)?;
                if node.rhs != 0 {
                    self.fn_id_stack.push(node_id);
                    self.check(node.rhs)?;
                    self.fn_id_stack.pop();
                }
            }
            Tag::Import => {}
            Tag::Prototype => {}
            Tag::Struct => {}
            Tag::VariableDecl => {
                let init_expr = self.tree.node_extra(node, 1);
                if init_expr != 0 {
                    self.check(init_expr)?;
                    // let annotation_id = self.tree.node_extra(node, 0);

                    let identifiers = self.tree.node(node.lhs);

                    let expressions = self.tree.node(init_expr);
                    let identifier_types: Vec<(NodeId, TypeId)> = self
                        .tree
                        .range(identifiers)
                        .map(|i| self.tree.node_index(i))
                        .map(|ni| (ni, self.get(ni)))
                        .collect();

                    let rtypes =
                        self.tree
                            .range(expressions)
                            .into_iter()
                            .fold(vec![], |mut v, i| {
                                let ti = self.get(self.tree.node_index(i));
                                let ti = self.flatten_var_type(ti);
                                match &self.types[ti] {
                                    Type::Tuple { fields } => v.extend(fields),
                                    _ => v.push(ti),
                                }
                                v
                            });

                    dbg!(&rtypes);

                    if self.tree.range(identifiers).len() != rtypes.len() {
                        return Err(Diagnostic::error()
                            .with_message(format!(
                                "Expected {} initial values, got {}.",
                                self.tree.range(identifiers).len(),
                                rtypes.len()
                            ))
                            .with_labels(vec![self.tree.label(node.token)]));
                    }

                    // rhs
                    for (i, (id, t)) in self.tree.range(expressions).zip(identifier_types) {
                        let expr_id = self.tree.node_index(i);
                        dbg!(self.get(expr_id));
                        // Coerce init expr into identifier type
                        let concrete = self.concretize(self.get(expr_id));
                        self.set(expr_id, concrete);
                        // if let Some(coerced) = self.coerce(expr_id, t) {
                        //     dbg!(t);
                        //     dbg!(id);
                        //     // Concretize identifier type
                        //     let concrete = self.concretize(id, t);
                        //     dbg!(concrete);
                        //     self.set_node_type(id, concrete);
                        //     self.tree.indices[i as usize] = coerced;
                        // }
                    }

                    // if annotation_id != 0 {
                    //     let annotation_type = self.get_node_type(annotation_id);
                    //     // Finalize lhs identifiers

                    //     let identifiers = self.tree.node(node.lhs);
                    //     for i in self.tree.range(identifiers) {
                    //         let ni = self.tree.node_index(i);
                    //         self.set_node_type(ni, annotation_type);
                    //     }
                    //     // Finalize rhs expressions
                    //     let expressions = self.tree.node(init_expr);
                    //     for i in self.tree.range(expressions) {
                    //         let ni = self.tree.node_index(i);
                    //         if let Some(coerced) = self.coerce(ni, annotation_type) {
                    //             self.tree.indices[i as usize] = coerced;
                    //         }
                    //     }
                    // } else {
                    //     // Finalize rhs expressions
                    //     let expressions = self.tree.node(init_expr);
                    //     let mut expr_types = Vec::<TypeId>::new();
                    //     for i in self.tree.range(expressions) {
                    //         let ni = self.tree.node_index(i);
                    //         let ti = self.get_node_type(ni);
                    //         let concrete_type = self.concretize(node_id, ti);
                    //         expr_types.extend(&self.flatten(concrete_type));
                    //     }
                    //     // dbg!(&expr_types);
                    //     let identifiers = self.tree.node(node.lhs);
                    //     for (index, i) in self.tree.range(identifiers).enumerate() {
                    //         let ni = self.tree.node_index(i);
                    //         self.set_node_type(ni, expr_types[index]);
                    //     }
                    // }

                    // let mut rtypes = Vec::<TypeId>::new();
                    // let expressions = self.tree.node(init_expr);
                    // for i in self.tree.range(expressions) {
                    //     let ni = self.tree.node_index(i);
                    //     let ti = self.get_node_type(ni);
                    //     // dbg!(&self.types[ti]);
                    //     match &self.types[ti] {
                    //         Type::Tuple { fields } => rtypes.extend(fields),
                    //         _ => rtypes.push(ti),
                    //     }
                    // }

                    // let identifiers = self.tree.node(node.lhs);
                }
            }
            // Statements
            Tag::Assign => {
                self.check(node.lhs)?;
                self.check(node.rhs)?;
                if let Some(coerced) = self.coerce(node.rhs, self.get(node.lhs)) {
                    self.tree.nodes[node_id as usize].rhs = coerced;
                } else {
                    return Err(Diagnostic::error()
                        .with_message(format!(
                            "mismatched types in assignment: expected {:?}, got {:?}",
                            self.types[self.get(node.lhs)],
                            self.types[self.get(node.rhs)]
                        ))
                        .with_labels(vec![self.tree.label(node.token)]));
                }
            }
            Tag::Return => {
                // let fn_id = self.fn_id_stack.last().unwrap();
                // let fn_type = self.get_t(*fn_id);
                // let fn_return_type = self.add_tuple_type(self.types[fn_type].returns().clone());
                for i in self.tree.range(node) {
                    let ni = self.tree.node_index(i);
                    self.check(ni)?;
                    // dbg!(&self.types[self.get_node_type(ni)]);
                    // dbg!(&self.types[fn_return_type]);
                    // Coerce value(s) into function return type.
                    // if let Some(child_node_id) = self.coerce(ni, fn_return_type) {
                    //     // dbg!(child_node_id);
                    //     self.tree.indices[i as usize] = child_node_id;
                    // } else {
                    //     return Err(Diagnostic::error()
                    //         .with_message(format!(
                    //             "incompatible return type: expected {:?}, got {:?}",
                    //             self.types[fn_return_type],
                    //             self.types[self.get_node_type(ni)],
                    //         ))
                    //         .with_labels(vec![self.tree.label(node.token)]));
                    // }
                }
            }
            Tag::If => {
                if node.lhs != 0 {
                    self.check(node.lhs)?;
                }
                if node.rhs != 0 {
                    self.check(node.rhs)?;
                }
            }
            Tag::While => {
                self.check(node.lhs)?;
                self.check(node.rhs)?;
            }
            // Expressions
            Tag::Access => {
                // Check lhs
                self.check(node.lhs)?;
                let concrete = self.concretize(self.get(node.lhs));
                self.set(node.lhs, concrete);

                // dbg!(self.get_t(node.lhs));

                if self.get(node.lhs) == T::Void as TypeId {
                    self.infer(node.rhs)?;
                    self.check(node.rhs)?;
                    // dbg!(self.get_t(node.rhs));
                    self.set(node_id, self.get(node.rhs));
                    return Ok(());
                }

                let mut ltype = self.get(node.lhs);
                while let Type::Pointer { typ, .. } = &self.types[ltype] {
                    ltype = *typ;
                }

                if let Some(&type_definition) = self.type_definitions.get(&ltype) {
                    let struct_decl = self.tree.node(type_definition);
                    assert_eq!(struct_decl.tag, Tag::Struct);
                    for i in self.tree.range(struct_decl) {
                        let field_id = self.tree.node_index(i);
                        if self.tree.name(field_id) == self.tree.name(node.rhs) {
                            self.definitions.insert(node_id, Definition::User(field_id));
                            let field = self.tree.node(field_id);
                            assert_eq!(field.tag, Tag::Field);
                            let field_index = self.tree.node_extra(field, 1);
                            self.definitions
                                .insert(node.rhs, Definition::User(field_index));
                            break;
                        }
                    }
                } else {
                    panic!();
                    // self.infer(node.rhs)?;
                    // self.check(node.rhs)?;
                    // return Ok(());
                }

                let container_type = &self.types[ltype];
                let rtype = match (container_type, self.definitions.get(&node.rhs)) {
                    (Type::Struct { fields, .. }, Some(Definition::User(field_index))) => {
                        fields[*field_index as usize]
                    }
                    // (Type::Pointer { typ, .. }, Some(Definition::User(field_index))) => {
                    //     if let Type::Struct { fields, .. } = se
                    // }
                    _ => {
                        return Err(Diagnostic::error()
                            .with_message(format!("type {:?} doesn't have fields", container_type))
                            .with_labels(vec![self.tree.label(node.token)]));
                    }
                };
                self.set(node_id, rtype);

                /*
                let mut ltype = self.infer_node(node.lhs)?;
                // Module access.
                if ltype == T::Void as TypeId {
                    return self.infer_node(node.rhs);
                }

                // Automatically dereference pointers.
                while let Type::Pointer { typ, .. } = &self.types[ltype] {
                    ltype = *typ;
                }

                // Resolve struct access based on types.
                if let Some(&type_definition) = self.type_definitions.get(&ltype) {
                    let struct_decl = self.tree.node(type_definition);
                    assert_eq!(struct_decl.tag, Tag::Struct);
                    for i in self.tree.range(struct_decl) {
                        let field_id = self.tree.node_index(i);
                        if self.tree.name(field_id) == self.tree.name(node.rhs) {
                            self.definitions.insert(node_id, Definition::User(field_id));
                            let field = self.tree.node(field_id);
                            assert_eq!(field.tag, Tag::Field);
                            let field_index = self.tree.node_extra(field, 1);
                            self.definitions
                                .insert(node.rhs, Definition::User(field_index));
                            break;
                        }
                    }
                }

                // Struct access.
                let container_type = &self.types[ltype];
                match (container_type, self.definitions.get(&node.rhs)) {
                    (Type::Struct { fields, .. }, Some(Definition::User(field_index))) => {
                        fields[*field_index as usize]
                    }
                    // (Type::Pointer { typ, .. }, Some(Definition::User(field_index))) => {
                    //     if let Type::Struct { fields, .. } = se
                    // }
                    _ => {
                        return Err(Diagnostic::error()
                            .with_message(format!("type {:?} doesn't have fields", container_type))
                            .with_labels(vec![self.tree.label(node.token)]));
                    }
                }
                */
            }
            Tag::Address => {
                self.check(node.lhs)?;
                // TODO: check for lvalue
            }
            Tag::Dereference => {
                self.check(node.lhs)?;
                // if let None = self.unify(self.get_t(node.lhs), Type::AnyPointer) {
                //     return Err(Diagnostic::error()
                //         .with_message(format!(
                //             "type \"{}\" cannot be dereferenced",
                //             self.format_type(pointer_type)
                //         ))
                //         .with_labels(vec![self.tree.label(self.tree.node(node.lhs).token)]));
                // }
            }
            Tag::Add | Tag::Mul => {
                // 1. Finalize arguments
                self.check(node.lhs)?;
                self.check(node.rhs)?;

                // 2. Infer callee
                let callee_id = node_id;

                // Figure out the definition the callee refers to based on the argument types.
                self.resolve_call(callee_id, &[node.lhs, node.rhs])?;

                // 3. Finalize callee

                let definition = *self.definitions.get(&callee_id).unwrap_or_else(|| {
                    panic!("Definition not found: {}", self.tree.name(callee_id))
                });

                let type_id = match definition {
                    Definition::User(decl_id) | Definition::Resolved(decl_id) => self.get(decl_id),
                    Definition::BuiltInFunction(built_in_fn) => self.add_function_type(
                        self.types[*self.builtin_function_types.get(&built_in_fn).unwrap()].clone(),
                    ),
                    _ => unreachable!(),
                };

                let parameter_types = self.types[type_id].parameters();
                assert_eq!(parameter_types.len(), 2);
                let lparam_type = parameter_types[0];
                let rparam_type = parameter_types[1];

                // dbg!(lparam_type);
                // dbg!(self.get_t(node.lhs));
                if let Some(coerced) = self.coerce(node.lhs, lparam_type) {
                    self.tree.nodes[node_id as usize].lhs = coerced;
                }
                if let Some(coerced) = self.coerce(node.rhs, rparam_type) {
                    self.tree.nodes[node_id as usize].rhs = coerced;
                }

                let type_id = if let Type::Function { returns, .. } = &self.types[type_id] {
                    if !returns.is_empty() {
                        self.add_tuple_type(returns.clone())
                    } else {
                        T::Void as TypeId
                    }
                } else {
                    // Callee type is not a function (e.g. a cast).
                    panic!();
                };
                // dbg!(type_id);
                // dbg!(self.get_node_type(node_id));
                // dbg!(&self.types[self.get_node_type(node_id)]);
                dbg!(self.unify(self.get(node_id), type_id));

                // self.set_node_type(node_id, type_id);
            }
            Tag::Div
            | Tag::Sub
            | Tag::BitwiseAnd
            | Tag::BitwiseOr
            | Tag::BitwiseShiftL
            | Tag::BitwiseShiftR
            | Tag::BitwiseXor => {
                self.check(node.lhs)?;
                self.check(node.rhs)?;
                dbg!(&self.types[self.get(node_id)]);
                dbg!(&self.types[self.get(node.lhs)]);
                dbg!(&self.types[self.get(node.rhs)]);
                if let Some(unified) = self.unify(self.get(node.lhs), self.get(node.rhs)) {
                    dbg!(unified);
                    dbg!(self.unify(unified, self.get(node_id)));
                    self.set(node.lhs, unified);
                    self.set(node.rhs, unified);
                    // self.set_t(node_id, unified);
                }
                // dbg!(&self.types[self.get_t(node_id)]);
                // if let Some(coerced) = self.coerce(node.lhs, self.get_t(node_id)) {
                //     self.tree.node_mut(node_id).lhs = coerced;
                // } else {
                //     return Err(Diagnostic::error()
                //         .with_message(format!(
                //             "incompatible operand {:?}, decl {:?}",
                //             &self.types[self.get_t(node.lhs)],
                //             &self.types[self.get_t(node_id)]
                //         ))
                //         .with_labels(vec![self.tree.label(self.tree.node(node.lhs).token)]));
                // }
                // dbg!(&self.types[self.get_t(node.rhs)]);
                // if let Some(coerced) = self.coerce(node.rhs, self.get_t(node_id)) {
                //     self.tree.node_mut(node_id).rhs = coerced;
                // } else {
                //     dbg!(&self.tree.node(node.rhs));
                //     return Err(Diagnostic::error()
                //         .with_message(format!(
                //             "incompatible operand {:?}, decl {:?}",
                //             &self.types[self.get_t(node.rhs)],
                //             &self.types[self.get_t(node_id)]
                //         ))
                //         .with_labels(vec![self.tree.label(self.tree.node(node.rhs).token)]));
                // }
            }
            Tag::Negation => {
                self.check(node.lhs)?;
                self.unify(self.get(node_id), self.get(node.lhs));
                if let Some(coerced) = self.coerce(node.lhs, self.get(node_id)) {
                    self.tree.node_mut(node_id).lhs = coerced;
                } else {
                    panic!();
                }
            }
            Tag::Equality | Tag::Inequality | Tag::LogicalAnd | Tag::LogicalOr => {
                self.check(node.lhs)?;
                self.check(node.rhs)?;
                // if let Some(coerced) = self.coerce(node.lhs, T::Boolean as TypeId) {
                //     self.tree.node_mut(node_id).lhs = coerced;
                // } else {
                //     panic!(
                //         "Failed to coerce logical expr to bool, got {:?}",
                //         self.get(node.lhs)
                //     );
                // }
                // if let Some(coerced) = self.coerce(node.rhs, T::Boolean as TypeId) {
                //     self.tree.node_mut(node_id).rhs = coerced;
                // } else {
                //     panic!();
                // }
            }
            Tag::Less => {
                self.check(node.lhs)?;
                self.check(node.rhs)?;
                if let Some(unified) = self.unify(self.get(node.lhs), self.get(node.rhs)) {
                    // if let Some(coerced) = self.coerce(node.lhs, unified) {
                    //     self.tree.node_mut(node_id).lhs = coerced;
                    // } else {
                    //     panic!();
                    // }
                    // if let Some(coerced) = self.coerce(node.rhs, unified) {
                    //     self.tree.node_mut(node_id).rhs = coerced;
                    // } else {
                    //     panic!();
                    // }
                }
            }
            Tag::Not => {
                self.check(node.lhs)?;
                if let Some(coerced) = self.coerce(node.lhs, T::Boolean as TypeId) {
                    self.tree.node_mut(node_id).lhs = coerced;
                } else {
                    panic!();
                }
            }
            Tag::Call => {
                // 1. Finalize arguments.
                self.check(node.rhs)?;

                // 2. Infer callee
                let callee_id = node.lhs;
                let expressions = self.tree.rchild(node);
                let argument_ids: Vec<NodeId> = self
                    .tree
                    .range(expressions)
                    .map(|i| self.tree.node_index(i))
                    .collect();
                self.resolve_call(callee_id, &argument_ids)?;
                self.infer(callee_id)?;

                // 3. Finalize callee
                self.check(callee_id)?;

                // 4. Finalize arguments.

                ////////////////////////////////////////////////////////////////

                let definition = *self.definitions.get(&callee_id).unwrap_or_else(|| {
                    panic!("Definition not found: {}", self.tree.name(callee_id))
                });

                dbg!(self.get(callee_id));
                dbg!(definition);

                let type_id = match definition {
                    Definition::User(decl_id) | Definition::Resolved(decl_id) => self.get(decl_id),
                    Definition::BuiltInFunction(built_in_fn) => self.add_function_type(
                        self.types[*self.builtin_function_types.get(&built_in_fn).unwrap()].clone(),
                    ),
                    Definition::BuiltIn(ty) => {
                        dbg!(self.unify(self.get(node_id), ty as TypeId));
                        return Ok(());
                    }
                    _ => unreachable!(),
                };

                dbg!(self.tree.node(callee_id).tag);
                assert_eq!(
                    type_id,
                    self.get(callee_id),
                    "{:?} {:?}",
                    &self.types[type_id],
                    &self.types[self.get(callee_id)]
                );
                // self.finalize(node.rhs)?;

                let parameter_types = self.types[type_id].parameters().clone();
                let arguments = self.tree.node(node.rhs);
                for (index, i) in self.tree.range(arguments).enumerate() {
                    let ni = self.tree.node_index(i);
                    // dbg!(parameter_types[index]);
                    if let Some(coerced) = self.coerce(ni, parameter_types[index]) {
                        self.tree.indices[i as usize] = coerced;
                    }
                }

                // 4. Set return type.
                let callee_type = self.get(callee_id);
                let type_id = if let Type::Function {
                    parameters: _,
                    returns,
                } = &self.types[callee_type]
                {
                    if !returns.is_empty() {
                        self.add_tuple_type(returns.clone())
                    } else {
                        T::Void as TypeId
                    }
                } else {
                    // Callee type is not a function (e.g. a cast).
                    callee_type as TypeId
                };
                // dbg!(type_id);
                dbg!(self.get(node_id));

                dbg!(self.unify(self.get(node_id), type_id));
                // self.set_node_type(node_id, type_id);
            }
            Tag::Subscript => {
                self.check(node.lhs)?;
                self.check(node.rhs)?;

                let element_type = self.types[self.get(node.lhs)].element_type();
                self.unify(self.get(node_id), element_type);

                if let Some(coerced) = self.coerce(node.rhs, T::I64 as TypeId) {
                    self.tree.node_mut(node_id).rhs = coerced;
                } else {
                    panic!();
                }

                // let var = self.new_unification_variable();
                // self.set(node_id, var);
            }
            // Simple expressions
            Tag::Identifier => {
                let decltype = self.get_identifier_type(node_id);
                let identifier_type = self.get(node_id);

                if is_none(decltype) && is_none(identifier_type) {
                    self.set(node_id, T::Void as TypeId);
                    return Ok(());
                }

                // if let Some(coerced) = self.coerce(node_id, identifier_t)
                if let Some(unified_type) = self.unify(identifier_type, decltype) {
                    println!(
                        "unifying {node_id}: {} vs {} -> {:?}",
                        identifier_type, decltype, unified_type
                    );
                    // self.set_node_type(node_id, unified_type);
                } else {
                    return Err(Diagnostic::error()
                        .with_message(format!(
                            "inference mismatch: identifier {:?}, decl {:?}",
                            &self.types[identifier_type], &self.types[decltype]
                        ))
                        .with_labels(vec![self.tree.label(node.token)]));
                }
            }
            // Literal expressions
            Tag::IntegerLiteral => {}
            Tag::FloatLiteral => {}
            Tag::True | Tag::False => {}
            Tag::StringLiteral => {}
            _ => {
                unreachable!("Unexpected node type in check: {:?}", node.tag);
            }
        }
        Ok(())
    }

    /// Coerces bound type variables into concrete types.
    fn finalize(&mut self, node_id: NodeId) -> Result<()> {
        let node = &self.tree.node(node_id).clone();
        println!("finalize: {node_id} {:?}", node.tag);
        match node.tag {
            Tag::Root | Tag::Module | Tag::Parameters | Tag::Block | Tag::IfElse => {
                for i in self.tree.range(node) {
                    let ni = self.tree.node_index(i);
                    self.finalize(ni)?;
                }
            }
            Tag::FunctionDecl => {
                if node.rhs != 0 {
                    self.fn_id_stack.push(node_id);
                    self.finalize(node.rhs)?;
                    self.fn_id_stack.pop();
                }
            }
            Tag::If => {
                if node.lhs != 0 {
                    self.finalize(node.lhs)?;
                }
                if node.rhs != 0 {
                    self.finalize(node.rhs)?;
                }
            }
            Tag::VariableDecl => {
                let init_expr = self.tree.node_extra(node, 1);

                let identifiers = self.tree.node(node.lhs);

                let expressions = self.tree.node(init_expr);
                let identifier_types: Vec<(NodeId, TypeId)> = self
                    .tree
                    .range(identifiers)
                    .map(|i| self.tree.node_index(i))
                    .map(|ni| (ni, self.get(ni)))
                    .collect();

                // rhs
                for (i, (id, t)) in self.tree.range(expressions).zip(identifier_types) {
                    let expr_id = self.tree.node_index(i);
                    dbg!(self.get(expr_id));
                    // Coerce init expr into identifier type
                    if let Some(coerced) = self.coerce(expr_id, t) {
                        dbg!(t);
                        dbg!(id);
                        // Concretize identifier type
                        let unified = self.unify(self.get(expr_id), t);
                        dbg!(unified);
                        let concrete = self.concretize(t);
                        dbg!(concrete);
                        self.set(id, concrete);
                        self.tree.indices[i as usize] = coerced;
                    }
                }
            }
            Tag::Return => {
                let fn_id = self.fn_id_stack.last().unwrap();
                let fn_type = self.get(*fn_id);
                let fn_return_type = self.add_tuple_type(self.types[fn_type].returns().clone());
                for i in self.tree.range(node) {
                    let ni = self.tree.node_index(i);
                    self.finalize(ni)?;
                    // dbg!(&self.types[self.get_node_type(ni)]);
                    // dbg!(&self.types[fn_return_type]);
                    // Coerce value(s) into function return type.
                    if let Some(child_node_id) = self.coerce(ni, fn_return_type) {
                        // dbg!(child_node_id);
                        self.tree.indices[i as usize] = child_node_id;
                    } else {
                        return Err(Diagnostic::error()
                            .with_message(format!(
                                "incompatible return type: expected {:?}, got {:?}",
                                self.types[fn_return_type],
                                self.types[self.get(ni)],
                            ))
                            .with_labels(vec![self.tree.label(node.token)]));
                    }
                }
            }
            Tag::Div
            | Tag::Sub
            | Tag::BitwiseAnd
            | Tag::BitwiseOr
            | Tag::BitwiseShiftL
            | Tag::BitwiseShiftR
            | Tag::BitwiseXor => {
                // dbg!(&self.types[self.get(node_id)]);
                if let Some(coerced) = self.coerce(node.lhs, self.get(node_id)) {
                    self.tree.node_mut(node_id).lhs = coerced;
                } else {
                    return Err(Diagnostic::error()
                        .with_message(format!(
                            "incompatible operand {:?}, decl {:?}",
                            &self.types[self.get(node.lhs)],
                            &self.types[self.get(node_id)]
                        ))
                        .with_labels(vec![self.tree.label(self.tree.node(node.lhs).token)]));
                }
                // dbg!(&self.types[self.get(node.rhs)]);
                if let Some(coerced) = self.coerce(node.rhs, self.get(node_id)) {
                    self.tree.node_mut(node_id).rhs = coerced;
                } else {
                    dbg!(&self.tree.node(node.rhs));
                    return Err(Diagnostic::error()
                        .with_message(format!(
                            "incompatible operand {:?}, decl {:?}",
                            &self.types[self.get(node.rhs)],
                            &self.types[self.get(node_id)]
                        ))
                        .with_labels(vec![self.tree.label(self.tree.node(node.rhs).token)]));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn get_identifier_type(&mut self, node_id: NodeId) -> TypeId {
        let decl = self.definitions.get(&node_id);
        if let Some(lookup) = decl {
            match lookup {
                Definition::User(decl_id) | Definition::Resolved(decl_id) => self.get(*decl_id),
                Definition::BuiltIn(type_id) => *type_id as TypeId,
                Definition::BuiltInFunction(built_in_fn) => self.add_function_type(
                    self.types[*self.builtin_function_types.get(built_in_fn).unwrap()].clone(),
                ),
                Definition::Foreign(_) => T::Void as TypeId,
                _ => panic!(),
            }
        } else {
            panic!(
                "Somehow failed to find identifier declaration for {:?}",
                self.tree.name(node_id)
            );
        }
    }

    fn new_type_variable(&mut self) -> TypeId {
        let id = self.next_variable_id;
        self.next_variable_id += 1;
        self.add_type(Type::Parameter {
            index: id,
            binding: T::Never as TypeId,
        })
    }

    fn coerce(&mut self, node_id: NodeId, t: TypeId) -> Option<NodeId> {
        // dbg!(self.get_node_type(node_id));
        // dbg!(&self.types[self.get(node_id)]);
        let node_type = self.concretize(self.get(node_id));
        println!(
            "  - coerce({node_id}: {} -> {node_type}, {t})",
            self.get(node_id)
        );
        self.set(node_id, node_type);
        if is_none(t) || !dbg!(self.is_subtype(node_type, t)) {
            return None;
        }
        // node_type <: t
        if dbg!(self.is_subtype_generic(node_type, t))
            || node_type == t
            || matches!(self.types[node_type], Type::Pointer { .. })
        {
            // println!("{node_type} {t}");
            return Some(node_id);
        }
        // dbg!(&self.types[self.get(node_id)]);
        // dbg!(&self.types[t]);
        let conversion_node_id =
            self.tree
                .add_node(Tag::Conversion, self.tree.node(node_id).token, node_id, 0);
        self.set(conversion_node_id, t);
        Some(conversion_node_id as NodeId)
    }

    fn concretize(&mut self, t: TypeId) -> TypeId {
        if is_none(t) || is(t, T::Void) {
            t
        } else {
            self.fullconcrete(t)
        }
    }

    fn fullconcrete(&mut self, t: TypeId) -> TypeId {
        if is_none(t) {
            panic!()
        }
        match self.types[t].clone() {
            Type::Numeric { .. } => {
                if is_integer_literal(t) {
                    t - 4
                } else {
                    t
                }
            }
            Type::Void | Type::None | Type::Boolean | Type::Never | Type::Struct { .. } => t,
            Type::Parameter { binding, .. } => {
                let t = self.fullconcrete(binding);
                if is(t, T::Void) {
                    panic!()
                } else {
                    t
                }
            }
            Type::Pointer { typ, .. } => {
                let t = self.fullconcrete(typ);
                if is(t, T::Void) {
                    panic!()
                } else {
                    self.add_pointer_type(t)
                }
            }
            Type::Tuple { fields } => {
                let mut concrete = vec![];
                for f in fields {
                    let c = self.fullconcrete(f);
                    concrete.push(c);
                }
                self.add_tuple_type(concrete)
            }
            Type::Function {
                parameters,
                returns,
            } => {
                let parameters = parameters.iter().map(|&t| self.fullconcrete(t)).collect();
                let returns = returns.iter().map(|&t| self.fullconcrete(t)).collect();
                self.add_function_type(Type::Function {
                    parameters,
                    returns,
                })
            }
            _ => unreachable!("Unexpected type! {:?}", &self.types[t]),
        }
    }

    fn flatten(&self, t: TypeId) -> Vec<TypeId> {
        if let Type::Tuple { fields } = &self.types[t] {
            fields.clone()
        } else {
            vec![t]
        }
    }

    /// Constrain types a and b.
    fn unify(&mut self, a: TypeId, b: TypeId) -> Option<TypeId> {
        println!(
            "  - unify({} : {:?}, {} : {:?})",
            a, &self.types[a], b, &self.types[b]
        );
        if is_none(a) || is_none(b) {
            return None;
        }
        if a == b {
            return Some(a);
        }

        // if let Type::Tuple { fields: a_fields } = self.types[a].clone() {
        //     if let Type::Tuple { fields: b_fields } = self.types[b].clone() {
        //         let mut u_fields = vec![];
        //         for (a, b) in a_fields.iter().zip(&b_fields) {
        //             if let Some(u) = self.unify(*a, *b) {
        //                 u_fields.push(u);
        //             } else {
        //                 return None;
        //             }
        //         }
        //         return Some(self.add_tuple_type(u_fields));
        //     } else {
        //         return None;
        //     }
        // }

        if self.is_subtype(b, a) {
            println!("b < a");
            return Some(a);
        }
        if self.is_subtype(a, b) {
            println!("a < b");
            return Some(b);
        }
        if let Type::Parameter { .. } = self.types[a] {
            if let Some(ty) = self.attempt_union(a, b) {
                return Some(ty);
            }
        } else if let Type::Parameter { .. } = self.types[b] {
            if let Some(ty) = self.attempt_union(b, a) {
                return Some(ty);
            }
        }
        None
    }

    /// Returns true if a is a subtype of b.
    fn is_subtype(&mut self, a: TypeId, b: TypeId) -> bool {
        println!("  - is_subtype({a}, {b})");
        if is_none(a) || is_none(b) {
            return false;
        }
        if self.is_subtype_generic(a, b) {
            // println!("is_subtype_generic: true");
            return true;
        }

        let b = self.flatten_var_type(b);

        match (self.types[a].clone(), self.types[b].clone()) {
            (Type::Never, _) | (Type::Any, _) => true,
            (
                Type::Numeric {
                    literal: a_literal,
                    floating: a_floating,
                    bytes: a_bytes,
                    ..
                },
                Type::Numeric {
                    literal: b_literal,
                    floating: b_floating,
                    bytes: b_bytes,
                    ..
                },
            ) => {
                if a_literal && !b_literal && !b_floating {
                    // No generic subtyping for constants.
                    false
                } else if a_floating {
                    // fLiteral <: fAny
                    // f8 <: f64
                    a_bytes <= b_bytes && b_floating
                } else {
                    // iLiteral <: iAny
                    // i8 <
                    // iAny <: fAny
                    a_bytes <= b_bytes || b_floating
                }
            }
            (Type::Numeric { .. }, _) => false,
            (Type::Pointer { typ: a_target, .. }, Type::Pointer { typ: b_target, .. }) => {
                self.is_subtype(a_target, b_target)
            }
            (Type::Pointer { .. }, _) => false,
            (Type::Tuple { fields: ref a }, Type::Tuple { fields: ref b }) => a
                .iter()
                .zip(b.iter())
                .all(|(ai, bi)| self.is_subtype(*ai, *bi)),
            _ => false,
        }
    }

    fn is_subtype_generic(&mut self, a: TypeId, b: TypeId) -> bool {
        println!("  - is_subtype_generic({a}, {b})");
        if is_none(a) || is_none(b) {
            return false;
        }
        if is(b, T::Any) {
            return true;
        }
        if a == b {
            return true;
        }
        let type_a = self.types[a].clone();
        let type_b = self.types[b].clone();
        if type_a == type_b {
            return true;
        }

        // Ensure the unification variable is on the left.
        if !matches!(type_a, Type::Parameter { .. }) && matches!(type_b, Type::Parameter { .. }) {
            return self.is_subtype(b, a);
        }

        // At this point, either both
        let b_original = b;
        // Make b concrete.
        let b = self.flatten_var_type(b);

        match (type_a, type_b) {
            (Type::Never, _) | (Type::Any, _) => true,
            (
                Type::Numeric {
                    literal: a_literal,
                    floating: a_floating,
                    bytes: a_bytes,
                    ..
                },
                Type::Numeric {
                    literal: b_literal,
                    floating: b_floating,
                    bytes: b_bytes,
                    ..
                },
            ) => {
                // dbg!(&type_a);
                // dbg!(&type_b);
                ((a_literal && a_bytes <= b_bytes) || (b_literal && b_bytes <= a_bytes))
                    && a_floating == b_floating
            }
            (Type::Numeric { .. }, _) => false,
            (Type::Pointer { typ: a_target, .. }, Type::Pointer { typ: b_target, .. }) => {
                self.is_subtype_generic(a_target, b_target)
            }
            (Type::Pointer { .. }, _) => false,
            (Type::Tuple { fields: a }, Type::Tuple { fields: b }) => a
                .iter()
                .zip(b.iter())
                .all(|(&ai, &bi)| self.is_subtype_generic(ai, bi)),
            (Type::Parameter { binding, .. }, _) if self.is_subtype(binding, b) => {
                // dbg!(binding);
                // dbg!(&self.types[b]);
                // dbg!(&self.types[b_original]);
                // Is a.binding <: b?
                // a.binding is more generic than b, and b does not occur in a
                if dbg!(self.more_generic(binding, b_original)) && !self.occurs(b_original, a) {
                    // Set a's binding to b
                    if let Type::Parameter {
                        ref mut binding, ..
                    } = &mut self.types[a]
                    {
                        dbg!(*binding);
                        dbg!(b_original);
                        *binding = b_original;
                    }
                } else if matches!(&self.types[b_original], Type::Parameter { .. }) {
                    panic!();
                    return self.is_subtype(b_original, a);
                }
                // a.binding is less generic than b
                true
            }
            // a is a parameter and so is b
            (Type::Parameter { .. }, Type::Parameter { .. }) => self.is_subtype(b_original, a),
            (Type::Parameter { .. }, _) => false,
            _ => false,
        }
    }

    fn flatten_var_type(&self, t: TypeId) -> TypeId {
        let mut tc = t;
        while let Type::Parameter { binding, .. } = self.types[tc] {
            tc = binding;
        }
        tc
    }

    fn attempt_union(&mut self, a: TypeId, b: TypeId) -> TypeRef {
        panic!();
        println!("  - attempt_union({a}, {b})");
        let mut ac = a;
        while let Type::Parameter { binding, .. } = self.types[ac] {
            // Replaces slot in list of types with copy of binding.
            // + no need to update references
            // - identical types may now have different indices
            self.types[ac] = self.types[binding].clone();

            // for node in self.tree.nodes.iter_mut() {
            //     if node.ty == a {
            //         node.ty = binding
            //     }
            // }

            // ac = binding;
        }
        let mut bc = b;
        while let Type::Parameter { binding, .. } = self.types[bc] {
            self.types[bc] = self.types[binding].clone();

            // for node in self.tree.nodes.iter_mut() {
            //     if node.ty == b {
            //         node.ty = binding
            //     }
            // }

            // bc = binding;
        }
        None
    }

    /// Returns true if a is more generic than b.
    fn more_generic(&self, a: TypeId, b: TypeId) -> bool {
        println!("  - more_generic({a}, {b})");
        if is(b, T::Never) || is(b, T::Any) {
            // Never and Any are the most generic types.
            return false;
        }

        if self.is_concrete(a)
            && !is(a, T::Never)
            && matches!(self.types[b], Type::Parameter { .. })
            && (!self.is_concrete(self.types[b].binding()) || is(self.types[b].binding(), T::Never))
        {
            // A is concrete and b is not
            return false;
        }
        dbg!(is_integer(a));
        dbg!(is_integer_literal(b));
        if is_integer(a) && is_integer_literal(b) {
            return false;
        }
        true
    }

    /// Check if the type t occurs in the type variable var.
    fn occurs(&self, t: TypeId, var: TypeId) -> bool {
        if t == var {
            return true;
        }
        match &self.types[t] {
            Type::Parameter { binding, .. } => self.occurs(*binding, var),
            _ => false,
        }
    }

    fn is_concrete(&self, t: TypeId) -> bool {
        match &self.types[t] {
            Type::Parameter { .. } | Type::Any => false,
            Type::Pointer { typ, .. } => self.is_concrete(*typ),
            _ => true,
        }
    }

    // fn fold_type(&mut self, fold: Fold) -> TypeId {
    //     match fold {
    //         Fold::None => T::None as TypeId,
    //         Fold::Integer(x) => {
    //             let var = self.new_unification_variable();
    //             self.unify(var, smallest_integer_type(x)).unwrap()
    //         }
    //         Fold::Float(_) => {
    //             let var = self.new_unification_variable();
    //             self.unify(var, T::F64 as TypeId).unwrap()
    //         }
    //         Fold::Boolean(_) => T::Boolean as TypeId,
    //     }
    // }

    // fn fold(&mut self, node_id: NodeId) -> Fold {
    //     use Fold::*;
    //     let node = self.tree.node(node_id).clone();
    //     match node.tag {
    //         Tag::IntegerLiteral => Integer(self.tree.node_lexeme(node_id).parse::<i64>().unwrap()),
    //         Tag::True => Boolean(true),
    //         Tag::False => Boolean(false),
    //         Tag::Add => {
    //             let (mut l, mut r) = (self.fold(node.lhs), self.fold(node.rhs));
    //             Fold::unify(&mut l, &mut r);
    //             match (l, r) {
    //                 (Integer(a), Integer(b)) => Integer(a + b),
    //                 (Float(a), Float(b)) => Float(a + b),
    //                 _ => None,
    //             }
    //         }
    //         _ => None,
    //     }
    // }

    ///
    fn infer_range(&mut self, node: &Node) -> Result<TypeId> {
        self.infer_range_with_types(node, None)
    }

    ///
    fn infer_range_with_types(
        &mut self,
        node: &Node,
        parent_type_ids: Option<Vec<TypeId>>,
    ) -> Result<TypeId> {
        let mut types = vec![];
        if let Some(type_ids) = parent_type_ids {
            for (index, i) in (node.lhs..node.rhs).enumerate() {
                let result = self.infer_node_with_type(
                    self.tree.node_index(i),
                    if type_ids.is_empty() {
                        None
                    } else {
                        Some(&type_ids[index])
                    },
                );
                if let Err(diagnostic) = result {
                    self.workspace.diagnostics.push(diagnostic);
                } else {
                    types.push(result.unwrap());
                }
            }
        } else {
            for i in node.lhs..node.rhs {
                let result = self.infer_node(self.tree.node_index(i));
                if let Err(diagnostic) = result {
                    self.workspace.diagnostics.push(diagnostic);
                } else {
                    types.push(result.unwrap())
                }
            }
        }
        match types.len() {
            0 => Ok(T::Void as TypeId),
            1 => Ok(types[0]),
            _ => Ok(self.add_tuple_type(types)),
        }
    }

    fn infer_node(&mut self, node_id: NodeId) -> Result<TypeId> {
        self.infer_node_with_type(node_id, None)
    }

    ///
    fn infer_node_with_type(
        &mut self,
        node_id: NodeId,
        parent_type_id: Option<&TypeId>,
    ) -> Result<TypeId> {
        if node_id == 0 {
            return Ok(T::Void as TypeId);
        }
        let node = &self.tree.node(node_id).clone();
        // println!("[{}] - {:?}", node_id, node.tag);
        let current_type_id = self.get(node_id);
        if current_type_id != T::Void as TypeId {
            return Ok(current_type_id);
        }
        let mut result: TypeId = match dbg!(node.tag) {
            Tag::Access => {
                let mut ltype = self.infer_node(node.lhs)?;
                // Module access.
                if ltype == T::Void as TypeId {
                    return self.infer_node(node.rhs);
                }

                // Automatically dereference pointers.
                while let Type::Pointer { typ, .. } = &self.types[ltype] {
                    ltype = *typ;
                }

                // Resolve struct access based on types.
                if let Some(&type_definition) = self.type_definitions.get(&ltype) {
                    let struct_decl = self.tree.node(type_definition);
                    assert_eq!(struct_decl.tag, Tag::Struct);
                    for i in self.tree.range(struct_decl) {
                        let field_id = self.tree.node_index(i);
                        if self.tree.name(field_id) == self.tree.name(node.rhs) {
                            self.definitions.insert(node_id, Definition::User(field_id));
                            let field = self.tree.node(field_id);
                            assert_eq!(field.tag, Tag::Field);
                            let field_index = self.tree.node_extra(field, 1);
                            self.definitions
                                .insert(node.rhs, Definition::User(field_index));
                            break;
                        }
                    }
                }

                // Struct access.
                let container_type = &self.types[ltype];
                match (container_type, self.definitions.get(&node.rhs)) {
                    (Type::Struct { fields, .. }, Some(Definition::User(field_index))) => {
                        fields[*field_index as usize]
                    }
                    // (Type::Pointer { typ, .. }, Some(Definition::User(field_index))) => {
                    //     if let Type::Struct { fields, .. } = se
                    // }
                    _ => {
                        return Err(Diagnostic::error()
                            .with_message(format!("type {:?} doesn't have fields", container_type))
                            .with_labels(vec![self.tree.label(node.token)]));
                    }
                }
            }
            Tag::Address => {
                let value_type = self.infer_node(node.lhs)?;
                self.add_pointer_type(value_type)
            }
            Tag::Add | Tag::Mul => {
                let callee_id = node_id;
                self.resolve_call(callee_id, &[node.lhs, node.rhs])?;

                // Get the newly resolved definition and type.
                let definition = *self.definitions.get(&callee_id).unwrap_or_else(|| {
                    panic!("Definition not found: {}", self.tree.name(callee_id))
                });

                let type_id = match definition {
                    Definition::User(decl_id) | Definition::Resolved(decl_id) => {
                        self.infer_node(decl_id)?
                    }
                    Definition::BuiltInFunction(built_in_fn) => self.add_function_type(
                        self.types[*self.builtin_function_types.get(&built_in_fn).unwrap()].clone(),
                    ),
                    _ => unreachable!(),
                };

                if let Type::Function {
                    parameters: _,
                    returns,
                } = &self.types[type_id]
                {
                    if !returns.is_empty() {
                        self.add_tuple_type(returns.clone())
                    } else {
                        T::Void as TypeId
                    }
                } else {
                    T::Void as TypeId
                }
            }
            Tag::Assign => {
                let ltype = self.infer_node(node.lhs)?;
                let rtype = self.infer_node_with_type(node.rhs, Some(&ltype))?;
                if node.rhs != 0 && ltype != rtype {
                    return Err(Diagnostic::error()
                        .with_message(format!(
                            "mismatched types in assignment: expected {:?}, got {:?}",
                            self.types[ltype], self.types[rtype]
                        ))
                        .with_labels(vec![self.tree.label(node.token)]));
                }
                T::Void as TypeId
            }
            Tag::Call => {
                // Argument expressions
                self.infer_node(node.rhs)?;
                let expressions = self.tree.rchild(node);

                // Callee
                let callee_id = node.lhs;
                let argument_ids: Vec<NodeId> = self
                    .tree
                    .range(expressions)
                    .map(|i| self.tree.node_index(i))
                    .collect();
                self.resolve_call(callee_id, &argument_ids)?;

                let ltype = self.infer_node(callee_id)?;

                if let Type::Function {
                    parameters: _,
                    returns,
                } = &self.types[ltype]
                {
                    if !returns.is_empty() {
                        self.add_tuple_type(returns.clone())
                    } else {
                        T::Void as TypeId
                    }
                } else {
                    // Callee type is not a function (e.g. a cast).
                    ltype as TypeId
                }
            }
            Tag::Dereference => {
                let pointer_type = self.infer_node(node.lhs)?;
                if let Type::Pointer { typ, .. } = self.types[pointer_type] {
                    typ
                } else {
                    return Err(Diagnostic::error()
                        .with_message(format!(
                            "type \"{}\" cannot be dereferenced",
                            self.format_type(pointer_type)
                        ))
                        .with_labels(vec![self.tree.label(self.tree.node(node.lhs).token)]));
                }
            }
            Tag::Field => self.infer_node(self.tree.node_index(node.rhs))?,
            Tag::FunctionDecl => {
                // Prototype
                let fn_type = self.infer_node(node.lhs)?;
                let fn_type_id = fn_type;
                // Immediately set the declaration's type to handle recursion.
                self.set(node_id, fn_type);
                // Body
                if self.current_fn_type_id.is_none() {
                    self.current_fn_type_id = Some(fn_type_id);
                }
                self.infer_node(node.rhs)?;
                self.current_fn_type_id = None;
                fn_type
            }
            Tag::Equality
            | Tag::Greater
            | Tag::GreaterEqual
            | Tag::Inequality
            | Tag::Less
            | Tag::LessEqual
            | Tag::LogicalAnd
            | Tag::LogicalOr => {
                let ltype = self.infer_node(node.lhs)?;
                let rtype = self.infer_node(node.rhs)?;
                if ltype != rtype {
                    if is_integer(ltype) && rtype == T::IntegerLiteral as TypeId {
                        self.set(node.rhs, ltype);
                        return Ok(ltype);
                    } else if ltype == T::IntegerLiteral as TypeId && is_integer(rtype) {
                        self.set(node.lhs, rtype);
                        return Ok(rtype);
                    }
                    return Err(Diagnostic::error()
                        .with_message(format!(
                            "mismatched types: left is \"{:?}\", right is \"{:?}\"",
                            self.types[ltype], self.types[rtype]
                        ))
                        .with_labels(vec![self.tree.label(node.token)]));
                } else if ltype == T::IntegerLiteral as TypeId
                    && rtype == T::IntegerLiteral as TypeId
                {
                    self.set(node.lhs, T::I64 as TypeId);
                    self.set(node.rhs, T::I64 as TypeId);
                }
                T::Boolean as TypeId
            }
            Tag::Identifier => {
                // The type of an identifier is the type of its definition.
                let decl = self.definitions.get(&node_id);
                if let Some(lookup) = decl {
                    match lookup {
                        Definition::User(decl_id) | Definition::Resolved(decl_id) => {
                            let decl_node = self.tree.node(*decl_id);
                            if decl_node.tag == Tag::FunctionDecl {
                                // Grab the prototype type without checking the function body again.
                                self.infer_node(decl_node.lhs)?
                            } else {
                                self.infer_node(*decl_id)?
                            }
                        }
                        Definition::BuiltIn(type_id) => *type_id as TypeId,
                        Definition::BuiltInFunction(built_in_fn) => self.add_function_type(
                            self.types[*self.builtin_function_types.get(built_in_fn).unwrap()]
                                .clone(),
                        ),

                        Definition::Overload(_) => unreachable!(),
                        // Definition::Foreign(_) => {
                        //     let ptr_type = self.add_pointer_type(2);
                        //     self.add_function_type(Type::Function {
                        //         parameters: vec![],
                        //         returns: vec![ptr_type],
                        //     })
                        // }
                        _ => T::Void as TypeId,
                    }
                } else {
                    T::Void as TypeId
                }
            }
            Tag::Prototype => {
                let mut parameters = Vec::new();
                let mut returns = Vec::new();

                let type_parameters_id = node.lhs;
                if type_parameters_id != 0 {
                    // Type parameters
                    let type_parameters = self.tree.node(type_parameters_id);
                    let lhs = type_parameters.lhs;
                    for i in lhs..type_parameters.rhs {
                        let node_id = self.tree.node_index(i);
                        self.add_type(Type::Parameter {
                            index: (i - lhs) as usize,
                            binding: T::Void as TypeId,
                        });
                        self.set(node_id, self.types.len() - 1);
                    }
                }

                let parameters_id = self.tree.node_extra(node, 0);
                self.infer_node(parameters_id)?; // parameters

                let returns_id = self.tree.node_extra(node, 1);
                let return_type = self.infer_node(returns_id)?; // returns

                let params = self.tree.node(parameters_id);
                let rets = self.tree.node(returns_id);

                assert_eq!(params.tag, Tag::Parameters);
                for i in params.lhs..params.rhs {
                    let ni = self.tree.node_index(i);
                    parameters.push(self.get(ni));
                }
                if rets.tag == Tag::Expressions {
                    for i in rets.lhs..rets.rhs {
                        let ni = self.tree.node_index(i);
                        returns.push(self.get(ni));
                    }
                } else if rets.tag == Tag::Identifier
                    || rets.tag == Tag::Type && return_type != T::Void as TypeId
                {
                    returns.push(return_type);
                }
                self.add_function_type(Type::Function {
                    parameters,
                    returns,
                })
            }
            Tag::Return => {
                let fn_type = &self.types[self.current_fn_type_id.unwrap()].clone();
                let return_types = fn_type.returns();
                let expr_type_id = self.infer_range_with_types(node, Some(return_types.clone()))?;
                if let Type::Tuple { fields } = &self.types[expr_type_id] {
                    if fields.len() != fn_type.returns().len() {
                        return Err(Diagnostic::error()
                            .with_message(format!(
                                "invalid number of return values: expected \"{:?}\", got \"{:?}\"",
                                fn_type.returns().len(),
                                fields.len()
                            ))
                            .with_labels(vec![self.tree.label(node.token)]));
                    } else {
                        for (index, ti) in return_types.iter().enumerate() {
                            if fields[index] != *ti {
                                return Err(Diagnostic::error()
                                    .with_message(format!(
                                        "mismatched types: expected type #{} to be \"{:?}\", got \"{:?}\"",
                                        index+1, self.types[*ti], self.types[fields[index]]
                                    ))
                                    .with_labels(vec![self.tree.label(node.token)]));
                            }
                        }
                    }
                } else if return_types.len() > 1 {
                    return Err(Diagnostic::error()
                        .with_message(format!(
                            "invalid number of return values: expected \"{:?}\", got \"{:?}\"",
                            return_types.len(),
                            1
                        ))
                        .with_labels(vec![self.tree.label(node.token)]));
                } else if return_types.len() == 1 && return_types[0] != expr_type_id {
                    return Err(Diagnostic::error()
                        .with_message(format!(
                            "mismatched types: expected \"{:?}\", got \"{:?}\"",
                            self.types[return_types[0]], self.types[expr_type_id]
                        ))
                        .with_labels(vec![self.tree.label(node.token)]));
                }

                T::Void as TypeId
            }
            Tag::Struct => {
                if node.lhs != 0 {
                    let type_parameters = self.tree.node(node.lhs);
                    let lhs = type_parameters.lhs;
                    for i in lhs..type_parameters.rhs {
                        let node_id = self.tree.node_index(i);
                        let param_type = self.add_type(Type::Parameter {
                            index: (i - lhs) as usize,
                            binding: T::Void as TypeId,
                        });
                        self.set(node_id, param_type);
                    }
                    self.polymorphic_types
                        .try_insert(node_id, HashMap::new())
                        .expect("tried to insert duplicate polymorphic type map");
                };
                // Infer field types
                let mut fields = Vec::new();
                for i in self.tree.range(node) {
                    self.infer_node(self.tree.node_index(i))?;
                }
                // Set struct type based on field types
                for i in self.tree.range(node) {
                    let ni = self.tree.node_index(i);
                    fields.push(self.get(ni));
                }
                let is_generic = fields.iter().any(|&type_id| self.is_generic(type_id));
                let struct_type = Type::Struct { fields, is_generic };
                let type_id = if self.tree.name(node_id) == "String" {
                    self.types[T::String as TypeId] = struct_type;
                    T::String as TypeId
                } else {
                    self.add_type(struct_type)
                };
                self.type_definitions.insert(type_id, node_id);
                type_id
            }
            Tag::StringLiteral => T::String as TypeId,
            Tag::Subscript => {
                let array_type = self.infer_node(node.lhs)?;
                self.infer_node_with_type(node.rhs, Some(&(T::I64 as TypeId)))?;
                if let Type::Array { typ, .. } = self.types[array_type] {
                    typ
                } else {
                    return Err(Diagnostic::error()
                        .with_message(format!(
                            "cannot index into a value of type \"{}\"",
                            self.format_type(array_type)
                        ))
                        .with_labels(vec![self.tree.label(self.tree.node(node.lhs).token)]));
                }
            }
            Tag::Type => {
                let definition = self.definitions.get(&node_id).unwrap();
                match definition {
                    Definition::BuiltIn(builtin) => match builtin {
                        T::Array => {
                            // Map concrete type arguments to an array type.
                            // Expect two type parameters.
                            assert_eq!(node.tag, Tag::Type);
                            if node.rhs - node.lhs != 2 {
                                return Err(Diagnostic::error()
                                    .with_message(format!(
                                        "Expected 2 type parameters, got {}.",
                                        node.rhs - node.lhs
                                    ))
                                    .with_labels(vec![self.tree.label(node.token)]));
                            }
                            let ni = self.tree.node_index(node.lhs);
                            let value_type = self.infer_node(ni)?;
                            let ni = self.tree.node_index(node.lhs + 1);
                            let length_node = self.tree.node(ni);
                            if length_node.tag != Tag::IntegerLiteral {
                                return Err(Diagnostic::error()
                                    .with_message(
                                        "The length of an Array must be an integer literal.",
                                    )
                                    .with_labels(vec![self.tree.label(length_node.token)]));
                            }
                            let token_str = self.tree.node_lexeme(ni);
                            let length = token_str.parse::<i64>().unwrap();
                            self.add_array_type(value_type, length as usize)
                        }
                        T::Pointer => {
                            // Map concrete type argument to a pointer type.
                            // Expect one type parameter
                            if node.rhs - node.lhs != 1 {
                                return Err(Diagnostic::error()
                                    .with_message(format!(
                                        "Expected 1 type parameter, got {}.",
                                        node.rhs - node.lhs
                                    ))
                                    .with_labels(vec![self.tree.label(node.token)]));
                            }
                            let ni = self.tree.node_index(node.lhs);
                            let value_type = self.infer_node(ni)?;
                            self.add_pointer_type(value_type)
                        }
                        _ => {
                            return Err(Diagnostic::error()
                                .with_message("Undefined built-in type")
                                .with_labels(vec![self.tree.label(node.token)]))
                        }
                    },
                    Definition::User(id) => {
                        // Map concrete type arguments to a struct type.
                        let struct_decl_id = *id;
                        self.current_struct_id = *id;
                        let current_parameter_count = self.current_parameters.len();
                        for i in self.tree.range(node) {
                            let ni = self.tree.node_index(i);
                            let param_type = self.infer_node(ni)?;
                            self.current_parameters.push(param_type);
                        }
                        let struct_type_id = self.get(struct_decl_id);
                        let specified_type = self.monomorphize_type(struct_type_id);
                        self.current_parameters.truncate(current_parameter_count);
                        self.current_struct_id = 0;
                        specified_type
                    }
                    _ => return Err(Diagnostic::error().with_message("Undefined type")),
                }
            }
            Tag::VariableDecl => {
                let identifiers = &self.tree.node(node.lhs).clone();
                let annotation_id = self.tree.node_extra(node, 0);
                let rvalues_id = self.tree.node_extra(node, 1);

                let annotation = self.infer_node(annotation_id)?;
                if rvalues_id == 0 {
                    // Set lhs types.
                    for i in self.tree.range(identifiers) {
                        let ni = self.tree.node_index(i);
                        self.set(ni, annotation);
                    }
                } else {
                    // Set rhs types.
                    self.infer_node_with_type(
                        rvalues_id,
                        if annotation == T::Void as TypeId {
                            None
                        } else {
                            Some(&annotation)
                        },
                    )?;

                    let mut rtypes = Vec::<TypeId>::new();
                    let rvalues = self.tree.node(rvalues_id);
                    for i in self.tree.range(rvalues) {
                        let ni = self.tree.node_index(i);
                        let ti = self.get(ni);
                        match &self.types[ti] {
                            Type::Tuple { fields } => rtypes.extend(fields),
                            _ => rtypes.push(ti),
                        }
                        if ti == T::IntegerLiteral as TypeId {
                            // let inferred_type = infer_type(annotation, ti)?;
                            // self.set(ni, inferred_type)
                        }
                    }

                    if self.tree.range(identifiers).len() != rtypes.len() {
                        return Err(Diagnostic::error()
                            .with_message(format!(
                                "Expected {} initial values, got {}.",
                                self.tree.range(identifiers).len(),
                                rtypes.len()
                            ))
                            .with_labels(vec![self.tree.label(node.token)]));
                    }

                    // Set lhs types.
                    for (i, ti) in self.tree.range(identifiers).zip(rtypes.iter()) {
                        let ni = self.tree.node_index(i);
                        // let inferred_type = infer_type(annotation, *ti)?;
                        // self.set(ni, inferred_type);
                    }

                    let tuple_type = self.add_tuple_type(rtypes);
                    self.set(rvalues_id, tuple_type);
                }
                T::Void as TypeId
            }
            _ => {
                self.infer_node(node.lhs)?;
                self.infer_node(node.rhs)?;
                T::Void as TypeId
            }
        };

        if let Type::Tuple { fields } = &self.types[result] {
            if fields.len() == 1 {
                result = fields[0];
            }
        }

        // After recursively calling inferring the types of child nodes, set the type of the parent.
        self.set(node_id, result);
        Ok(result)
    }

    fn resolve_call(&mut self, callee_id: NodeId, argument_ids: &[NodeId]) -> Result<()> {
        // The callee may be:
        // - an unresolved overload (which may include user-defined, or built-in functions)
        // - a user-defined function
        // - a built-in function
        // - a built-in type (for)
        let definition = *self
            .definitions
            .get(&callee_id)
            .unwrap_or_else(|| panic!("Definition not found: {}", self.tree.name(callee_id)));
        match definition {
            Definition::Overload(overload_set_id) => {
                let overload_set = self.overload_sets.get(&overload_set_id).unwrap();
                'outer: for &definition in overload_set {
                    match definition {
                        Definition::User(fn_decl_id) => {
                            let resolution = Definition::Resolved(fn_decl_id);
                            let fn_decl = self.tree.node(fn_decl_id);
                            let fn_type_id = self.get(fn_decl.lhs);
                            // If the arguments could match the parameters
                            if self.check_arguments(fn_type_id, argument_ids).is_ok() {
                                self.definitions.insert(callee_id, resolution);
                                return Ok(());
                            } else {
                                continue 'outer;
                            }
                        }
                        Definition::BuiltInFunction(built_in_function) => {
                            let fn_type_id =
                                *self.builtin_function_types.get(&built_in_function).unwrap();
                            if self.check_arguments(fn_type_id, argument_ids).is_ok() {
                                self.definitions.insert(callee_id, definition);
                                return Ok(());
                            } else {
                                continue 'outer;
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                return Err(Diagnostic::error()
                    .with_message("failed to find matching overload")
                    .with_labels(vec![self.tree.label(self.tree.node(callee_id).token)]));
            }
            Definition::User(definition_id) => {
                let fn_decl = self.tree.node(definition_id);
                let prototype = &self.tree.node(fn_decl.lhs).clone();
                match prototype.tag {
                    Tag::Prototype => {
                        if prototype.lhs != 0 {
                            // Parametric procedure
                            let mut flat_argument_types = vec![];
                            for node_id in argument_ids {
                                let t = self.get(*node_id);
                                if let Type::Tuple { fields } = &self.types[t] {
                                    flat_argument_types.extend(fields);
                                } else {
                                    // If the argument is an integer literal.
                                    // dbg!(&self.types[t]);
                                    if t == T::IntegerLiteral as TypeId {
                                        self.set(*node_id, T::I64 as TypeId);
                                        flat_argument_types.push(T::I64 as TypeId);
                                        continue;
                                    }
                                    flat_argument_types.push(t);
                                }
                            }
                            let type_parameters = self.tree.node(prototype.lhs);
                            let parameters = self.tree.node(self.tree.node_extra(prototype, 0));
                            let returns_id = self.tree.node_extra(prototype, 1);
                            let returns = self.tree.node(returns_id);
                            let mut type_arguments =
                                vec![T::None as TypeId; self.tree.range(type_parameters).count()];
                            for (pi, i) in self.tree.range(parameters).enumerate() {
                                let ni = self.tree.node_index(i);
                                let ti = self.get(ni);
                                let arg_type = flat_argument_types[pi];
                                if let Type::Parameter { index, .. } = self.types[ti] {
                                    // Parameter is generic, set type arguments based on  argument type.
                                    let param_type = type_arguments[index];
                                    if param_type == T::None as TypeId {
                                        type_arguments[index] = arg_type;
                                    } else if arg_type != param_type {
                                        return Err(Diagnostic::error().with_message(format!(
                                            "mismatched types in function call: expected {:?} argument, got {:?}",
                                            self.types[param_type],
                                            self.types[arg_type])
                                        ).with_labels(vec![self
                                            .tree
                                            .label(self.tree.node(callee_id).token)]));
                                    }
                                }
                            }
                            let mut return_type_ids = vec![];
                            if returns_id != 0 {
                                assert_eq!(returns.tag, Tag::Expressions);
                                for i in self.tree.range(returns) {
                                    let ni = self.tree.node_index(i);
                                    let ti = self.get(ni);
                                    let ti = match self.types[ti] {
                                        Type::Parameter { index, .. } => type_arguments[index],
                                        Type::Pointer { typ, .. } => {
                                            if let Type::Parameter { index, .. } = self.types[typ] {
                                                type_arguments[index]
                                            } else {
                                                ti
                                            }
                                        }
                                        _ => ti,
                                    };
                                    return_type_ids.push(ti);
                                }
                            }
                            if type_arguments.contains(&(T::None as TypeId)) {
                                panic!("failed to set all type arguments in function call");
                            }
                            if let Err(mut occupied_err) = self.type_parameters.try_insert(
                                definition_id,
                                HashMap::from([(
                                    flat_argument_types.clone(),
                                    type_arguments.clone(),
                                )]),
                            ) {
                                occupied_err
                                    .entry
                                    .get_mut()
                                    .insert(flat_argument_types.clone(), type_arguments);
                            }

                            let callee_type_id = self.add_function_type(Type::Function {
                                parameters: flat_argument_types.clone(),
                                returns: return_type_ids,
                            });
                            self.set(callee_id, callee_type_id);
                            return Ok(());
                        }

                        // Non-parametric procedure: just type-check the arguments.
                        let fn_type_id = self.get(fn_decl.lhs);
                        dbg!(&argument_ids);
                        dbg!(&self.types[fn_type_id]);
                        return self
                            .check_arguments(fn_type_id, argument_ids)
                            .map_err(|err| {
                                err.with_labels(vec![self
                                    .tree
                                    .label(self.tree.node(callee_id).token)])
                            });
                    }
                    _ => {
                        panic!()
                    }
                }
            }
            Definition::BuiltInFunction(built_in_function) => {
                let fn_type_id = *self.builtin_function_types.get(&built_in_function).unwrap();
                return self
                    .check_arguments(fn_type_id, argument_ids)
                    .map_err(|err| {
                        err.with_labels(vec![self.tree.label(self.tree.node(callee_id).token)])
                    });
            }
            Definition::Foreign(_) => {}
            Definition::BuiltIn(built_in_type) => self.set(callee_id, built_in_type as TypeId),
            _ => unreachable!("Definition not found: {}", self.tree.name(callee_id)),
        }
        Ok(())
    }

    ///  
    fn check_arguments(&mut self, fn_type_id: TypeId, argument_ids: &[NodeId]) -> Result<()> {
        let fn_type = &self.types[fn_type_id].clone();
        let parameter_types = fn_type.parameters();
        let mut parameter_index = 0;
        for node_id in argument_ids {
            let t = self.get(*node_id);
            if parameter_index >= parameter_types.len() {
                return Err(Diagnostic::error().with_message(format!(
                    "invalid function call: expected {:?} arguments, got {:?}",
                    parameter_types.len(),
                    argument_ids.len()
                )));
            }
            for &arg_type in &self.type_ids(t) {
                let param_type = parameter_types[parameter_index];
                parameter_index += 1;
                if !self.is_subtype(arg_type, param_type) {
                    return Err(Diagnostic::error().with_message(format!(
                        "mismatched types in function call: expected {:?}, got {:?}",
                        self.types[param_type], self.types[arg_type]
                    )));
                }
            }
        }
        if parameter_index != parameter_types.len() {
            return Err(Diagnostic::error().with_message(format!(
                "invalid function call: expected {:?} arguments, got {:?}",
                parameter_types.len(),
                argument_ids.len()
            )));
        }
        Ok(())
    }

    fn type_ids(&self, type_id: TypeId) -> Vec<TypeId> {
        match &self.types[type_id] {
            Type::Tuple { fields } => fields.clone(),
            _ => vec![type_id],
        }
    }

    fn add_type(&mut self, typ: Type) -> TypeId {
        self.types.push(typ);
        self.types.len() - 1
    }

    fn add_array_type(&mut self, value_type: TypeId, length: usize) -> TypeId {
        match self
            .array_types
            .try_insert((value_type, length), self.types.len())
        {
            Ok(_) => self.add_type(Type::Array {
                typ: value_type,
                length,
                is_generic: self.is_generic(value_type),
            }),
            Err(err) => *err.entry.get(),
        }
    }

    fn add_function_type(&mut self, function_type: Type) -> TypeId {
        if let Type::Function { .. } = function_type {
            match self
                .function_types
                .try_insert(function_type.clone(), self.types.len())
            {
                Ok(_) => self.add_type(function_type),
                Err(err) => *err.entry.get(),
            }
        } else {
            unreachable!()
        }
    }

    fn add_pointer_type(&mut self, value_type: TypeId) -> TypeId {
        match self.pointer_types.try_insert(value_type, self.types.len()) {
            Ok(_) => self.add_type(Type::Pointer {
                typ: value_type,
                is_generic: self.is_generic(value_type),
            }),
            Err(err) => *err.entry.get(),
        }
    }

    fn add_polymorphic_type(&mut self, struct_id: NodeId, fields: Vec<TypeId>) -> TypeId {
        let type_map = self.polymorphic_types.get_mut(&struct_id).unwrap();
        match type_map.try_insert(self.current_parameters.clone(), self.types.len()) {
            Ok(_) => {
                let is_generic = fields.iter().any(|&type_id| self.is_generic(type_id));
                let type_id = self.add_type(Type::Struct { fields, is_generic });
                self.type_definitions.insert(type_id, struct_id);
                type_id
            }
            Err(err) => *err.entry.get(),
        }
    }

    fn add_tuple_type(&mut self, field_types: Vec<TypeId>) -> TypeId {
        match field_types.len() {
            0 => T::Void as TypeId,
            1 => field_types[0],
            _ => match self
                .tuple_types
                .try_insert(field_types.clone(), self.types.len())
            {
                Ok(_) => self.add_type(Type::Tuple {
                    fields: field_types,
                }),
                Err(err) => *err.entry.get(),
            },
        }
    }

    /// Creates a type based on the provided type_id, with all type parameters replaced by concrete types.
    fn monomorphize_type(&mut self, type_id: TypeId) -> TypeId {
        let typ = self.types[type_id].clone();
        match typ {
            Type::Parameter { index, .. } => self.current_parameters[index],
            Type::Array {
                typ,
                length,
                is_generic,
            } => {
                let specific_type = if is_generic {
                    self.monomorphize_type(typ)
                } else {
                    typ
                };
                self.add_array_type(specific_type, length)
            }
            Type::Pointer { typ, is_generic } => {
                let specific_type = if is_generic {
                    self.monomorphize_type(typ)
                } else {
                    typ
                };
                self.add_pointer_type(specific_type)
            }
            Type::Struct { fields, is_generic } => {
                let specific_types = if is_generic {
                    fields
                        .iter()
                        .map(|&type_id| self.monomorphize_type(type_id))
                        .collect()
                } else {
                    fields
                };
                self.add_polymorphic_type(self.current_struct_id, specific_types)
            }
            _ => type_id,
        }
    }

    fn is_generic(&self, type_id: TypeId) -> bool {
        match self.types[type_id] {
            Type::Parameter { .. } => true,
            Type::Array { is_generic, .. }
            | Type::Pointer { is_generic, .. }
            | Type::Struct { is_generic, .. } => is_generic,
            _ => false,
        }
    }

    #[inline]
    fn get(&self, node_id: NodeId) -> TypeId {
        self.tree.node(node_id).ty
    }

    ///
    fn set(&mut self, node_id: NodeId, type_id: TypeId) {
        if let Type::Tuple { fields } = &self.types[type_id] {
            if fields.len() == 1 {
                self.tree.node_mut(node_id).ty = fields[0];
                return;
            }
        }
        self.tree.node_mut(node_id).ty = type_id;
    }

    ///
    pub fn print(&self) {
        for i in 0..self.types.len() {
            println!(
                "{:>4} {:<24} {}",
                format!("{}.", i),
                self.format_type(i),
                format_args!("{:?}", self.types[i]),
            );
        }
    }

    pub fn format_type(&self, type_id: TypeId) -> String {
        let typ = &self.types[type_id];
        match typ {
            Type::Pointer { typ, .. } => format!("Pointer{{{}}}", self.format_type(*typ)),
            Type::Array { typ, length, .. } => {
                format!("Array{{{}, {}}}", self.format_type(*typ), length)
            }
            Type::Parameter { .. } => "Parameter".to_string(),
            Type::Function { .. } | Type::Struct { .. } => {
                if let Some(decl_id) = self.type_definitions.get(&type_id) {
                    self.tree.name(*decl_id).to_string()
                } else {
                    "Built-in".to_string()
                }
            }
            _ => format!("{:?}", typ),
        }
    }
}

// pub enum Fold {
//     None,
//     Integer(i64),
//     Float(f64),
//     Boolean(bool),
// }

// impl Fold {
//     fn integer(x: i64) -> Self {
//         Fold::Integer(x)
//     }

//     fn unify(l: &mut Fold, r: &mut Fold) {
//         if std::mem::discriminant(l) == std::mem::discriminant(r) {
//             return;
//         }
//         match (&l, &r) {
//             (Fold::Float(_), Fold::Integer(x)) => {
//                 *r = Fold::Float(*x as f64);
//             }
//             (Fold::Integer(x), Fold::Float(_)) => {
//                 *l = Fold::Float(*x as f64);
//             }
//             _ => {
//                 *l = Fold::None;
//                 *r = Fold::None;
//             }
//         }
//     }
// }

fn is_integer(t: TypeId) -> bool {
    (T::I8 as TypeId..=T::I64 as TypeId).contains(&t)
        || (T::U8 as TypeId..=T::U64 as TypeId).contains(&t)
}

fn is_integer_literal(t: TypeId) -> bool {
    t >= T::CI8 as TypeId && t <= T::CI64 as TypeId
}

fn smallest_integer_type(x: i64) -> TypeId {
    (if (-128..=127).contains(&x) {
        T::CI8
    } else if (-32768..=32767).contains(&x) {
        T::CI16
    } else if (-2147483648..=2147483647).contains(&x) {
        T::CI32
    } else {
        T::CI64
    }) as TypeId
}

#[inline]
fn is_none(type_id: TypeId) -> bool {
    type_id == T::None as TypeId
}

#[inline]
fn is(type_id: TypeId, ty: T) -> bool {
    type_id == ty as TypeId
}
