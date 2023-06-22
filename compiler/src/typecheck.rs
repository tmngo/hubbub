use crate::{
    analyze::{BuiltInFunction, Definition},
    parse::{NodeId, Tag, Tree},
    types::{float_type, integer_literal_type, integer_type, Type, TypeId, T},
    workspace::{Result, Workspace},
};
use codespan_reporting::diagnostic::Diagnostic;
use std::{collections::HashMap, rc::Rc, time::Instant};

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
        types[T::Never as TypeId] = Type::Never;
        types[T::Void as TypeId] = Type::Void;
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
        types[T::Type as TypeId] = Type::Type {
            ty: T::Any as TypeId,
        };

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

        let mut builtin_function_types = HashMap::new();
        // Add
        let binary_int_op_type = Type::Function {
            parameters: Rc::new([T::I64 as TypeId, T::I64 as TypeId]),
            returns: vec![T::I64 as TypeId],
        };
        types.push(binary_int_op_type);
        let binary_int_op_type_id = types.len() - 1;
        // Add i8
        let add_i8_type = types.len();
        types.push(Type::Function {
            parameters: vec![T::I8 as TypeId, T::I8 as TypeId].into(),
            returns: vec![T::I8 as TypeId],
        });
        // sizeof
        types.push(Type::Function {
            parameters: vec![T::Any as TypeId].into(),
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

        let start = Instant::now();
        if let Err(diagnostic) = self.infer(0) {
            self.workspace.diagnostics.push(diagnostic);
            return;
        }
        let t_infer = start.elapsed();

        self.print();
        println!("{:#?}", self.tree);

        println!("{}", crate::format_red!("---- CHECK ----"));

        let start = Instant::now();
        if let Err(diagnostic) = self.check(0) {
            self.workspace.diagnostics.push(diagnostic);
            return;
        }
        let t_check = start.elapsed();

        self.print();
        println!("{:#?}", self.tree);

        println!("Infer    {:>9.1?} ms", t_infer.as_secs_f64() * 1000.0);
        println!("Check    {:>9.1?} ms", t_check.as_secs_f64() * 1000.0);
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
                self.infer(node.lhs).unwrap();
                // self.type_definitions.insert(type_id, node_id);
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
            Tag::Import => {}
            _ => {
                panic!("{:?}", node.tag);
            }
        }
        Ok(())
    }

    /// Computes a type for every node.
    /// If a type can't be inferred yet, create a type variable.
    // boolean literal: boolean
    // integer literal: iX
    // float literal: fX
    // assign: check RHS
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
            Tag::Import => {
                self.set(node_id, T::Void as TypeId);
            }
            Tag::Expressions => {
                let mut ts = vec![];
                for i in self.tree.range(node) {
                    ts.push(self.infer(self.tree.node_index(i))?);
                }
                let t = self.add_tuple_type(ts);
                self.set(node_id, t);
            }
            // Declarations
            Tag::FunctionDecl => {
                self.infer(node.lhs)?;
                let fn_type_id = self.get(node.lhs);
                self.type_definitions.insert(fn_type_id, node_id);
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
                        let param_type = self.add_type(Type::TypeParameter {
                            index: (i - start) as usize,
                            binding: T::None as TypeId,
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
                    parameters: parameters.into(),
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
                        let param_type = self.add_type(Type::TypeParameter {
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
                // Type information flows from annotation -> expressions -> variables
                //
                // variable : annotation
                //  - set variable to type of annotation
                // variable : annotation = expression
                //  - set variable to type of annotation
                //  - unify variable and expression types
                // variable := expression
                //  - create type variable for annotation
                //  - unify variable, annotation, and expression types

                // Detect
                let identifiers = &self.tree.node(node.lhs).clone();
                let annotation_id = self.tree.node_extra(node, 0);
                let expressions_id = self.tree.node_extra(node, 1);

                if annotation_id == 0 && expressions_id == 0 {
                    panic!("Somehow a variable declaration is missing both an annotation and initial values!")
                }

                // panic!("{}", annotation);
                let annotation_t = if annotation_id != 0 {
                    self.infer(annotation_id)?
                } else {
                    self.new_type_variable()
                };

                // Infer
                if expressions_id == 0 {
                    // let annotation_type = self.infer(annotation_id)?;
                    // Set lhs types
                    for i in self.tree.range(identifiers) {
                        let ni = self.tree.node_index(i);
                        self.set(ni, annotation_t);
                    }
                } else {
                    // Infer expression types
                    self.infer(expressions_id)?;

                    // Unify lhs and rhs types
                    let mut rtypes = Vec::<TypeId>::new();
                    let rvalues = self.tree.node(expressions_id);
                    for i in self.tree.range(rvalues) {
                        let ni = self.tree.node_index(i);
                        let ti = self.get(ni);
                        self.narrow(ti, annotation_t);
                        match &self.types[ti] {
                            Type::Tuple { fields } => rtypes.extend(fields),
                            _ => rtypes.push(ti),
                        }
                    }

                    // let annotation_type = if annotation_id == 0
                    //     && rtypes.iter().all(|rtype| is_integer_literal(*rtype))
                    // {
                    //     T::I64 as TypeId
                    // } else {
                    //     self.new_type_variable()
                    // };

                    // Set lhs types
                    for i in self.tree.range(identifiers) {
                        let ni = self.tree.node_index(i);
                        self.set(ni, annotation_t);
                    }

                    for (i, rtype) in self.tree.range(identifiers).zip(rtypes.iter()) {
                        let ni = self.tree.node_index(i);
                        let ltype = self.get(ni);
                        // self.set(ni, *rtype);
                        self.narrow(ltype, *rtype);
                        // if let Some(unified_type) = self.unify(*rtype, ltype) {
                        //     // Fixes subtraction, breaks boolean.hb
                        //     // self.set_node_type(ni, unified_type);
                        // } else {
                        //     panic!()
                        // }
                    }

                    // Set tuple type
                    // let tuple_type = self.add_tuple_type(rtypes);
                    // self.set_node_type(expressions_id, tuple_type);
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
                self.narrow(b, a);

                // dbg!(self.unify(b, a));
                // self.set_node_type(node.rhs, self.get_node_type(node.lhs));
                // self.is_subtype(b, a);
            }
            Tag::Return => {
                // Infer children
                let expr_type = if node.lhs == 0 {
                    T::Void as TypeId
                } else {
                    self.infer(node.lhs)?
                };
                if expr_type != 0 {
                    // Narrow
                    let node_id = *self.fn_id_stack.last().unwrap();
                    let fn_t = self.get(node_id);
                    let return_t = self.add_tuple_type(self.types[fn_t].returns().clone());
                    self.narrow(expr_type, return_t);

                    // Coerce
                    // let expr_list = self.tree.node(node.lhs);
                    // let fn_return_types = self.type_ids(fn_return_type);
                    // for (index, i) in self.tree.range(expr_list).enumerate() {
                    //     let ni = self.tree.node_index(i);
                    //     if let Some(coerced) = self.coerce(ni, fn_return_types[index]) {
                    //         self.tree.indices[i as usize] = coerced;
                    //     }
                    // }
                    // self.set(node.lhs, fn_return_type);
                }
            }
            // Operator expressions
            Tag::Access => {
                self.infer(node.lhs)?;
                // Use a type variable since we can't figure out the type yet
                let var = self.new_type_variable();
                self.set(node_id, var);
                // Cannot typecheck rhs until lhs is inferred and concrete.
            }
            Tag::Address => {
                self.infer(node.lhs)?;
                // At this point, we can infer that this is a pointer to something.
                let var = self.new_type_variable();
                self.narrow(var, self.get(node.lhs));
                // self.unify(var, self.get(node.lhs));
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
                // Use a type variable since we can't figure out the type yet
                let var = self.new_type_variable();
                self.set(node_id, var);
                // If the callee is unambiguous, narrow the argument types according to the parameter types.
                let callee_id = node.lhs;
                let definition = *self.definitions.get(&callee_id).unwrap_or_else(|| {
                    panic!("Definition not found: {}", self.tree.name(callee_id))
                });
                match definition {
                    Definition::User(definition_id) => {
                        // Unambiguous function call.
                        let fn_decl = self.tree.node(definition_id);
                        let prototype = self.tree.node(fn_decl.lhs);
                        if prototype.lhs == 0 {
                            let fn_t = self.get(fn_decl.lhs);

                            // Narrow argument types
                            let args_t = self.get(node.rhs);
                            let parameters_t =
                                self.add_tuple_type(self.types[fn_t].parameters().to_vec());
                            self.narrow(args_t, parameters_t);

                            // Narrow return types
                            let return_t = self.add_tuple_type(self.types[fn_t].returns().clone());
                            self.narrow(self.get(node_id), return_t);
                            // dbg!(return_t);
                        }
                    }
                    Definition::BuiltInFunction(func) => {
                        let fn_t = *self.builtin_function_types.get(&func).unwrap();

                        // Narrow argument types
                        let args_t = self.get(node.rhs);
                        let parameters_ti =
                            self.add_tuple_type(self.types[fn_t].parameters().to_vec());
                        self.narrow(args_t, parameters_ti);

                        // Narrow return types
                        let returns_ti = self.add_tuple_type(self.types[fn_t].returns().clone());
                        self.narrow(self.get(node_id), returns_ti);
                    }
                    Definition::BuiltInType(built_in_t) => {
                        // dbg!(&self.types[built_in_t as TypeId]);
                        // self.narrow(self.get(node.rhs), built_in_t as TypeId);
                        self.set(callee_id, built_in_t as TypeId);
                        self.narrow(self.get(node_id), built_in_t as TypeId);
                    }
                    Definition::Overload(overload_set_id) => {
                        let overload_set = self.overload_sets.get(&overload_set_id).unwrap();
                        let mut fn_type_ids = vec![];
                        let expressions = self.tree.node(node.rhs);
                        let argument_ids: Vec<NodeId> = self
                            .tree
                            .range(expressions)
                            .map(|i| self.tree.node_index(i))
                            .collect();
                        for &definition in overload_set {
                            match definition {
                                Definition::User(fn_decl_id) => {
                                    let fn_decl = self.tree.node(fn_decl_id);
                                    let fn_type_id = self.get(fn_decl.lhs);
                                    // If the arguments could match the parameters
                                    if self.check_arguments(fn_type_id, &argument_ids).is_ok() {
                                        fn_type_ids.push(fn_type_id);
                                    }
                                }
                                Definition::BuiltInFunction(built_in_function) => {
                                    let fn_type_id = *self
                                        .builtin_function_types
                                        .get(&built_in_function)
                                        .unwrap();
                                    if self.check_arguments(fn_type_id, &argument_ids).is_ok() {
                                        fn_type_ids.push(fn_type_id);
                                    }
                                }
                                _ => unreachable!(),
                            }
                        }
                        self.print();
                        // One overload found
                        if fn_type_ids.len() == 1 {
                            let fn_t = fn_type_ids[0];
                            let args_t = self.get(node.rhs);
                            let parameters_type_id =
                                self.add_tuple_type(self.types[fn_t].parameters().to_vec());
                            self.narrow(args_t, parameters_type_id);
                            let return_t = self.add_tuple_type(self.types[fn_t].returns().clone());
                            self.narrow(self.get(node_id), return_t);
                        }
                    }
                    _ => unreachable!(),
                }
                // If the caleee is overloaded, narrow the
            }
            Tag::Add | Tag::Mul => {
                self.infer(node.lhs)?;
                self.infer(node.rhs)?;
                // Use a type variable since we can't figure out the type yet
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
                let _rtype = self.infer(node.rhs)?;
                // assert_eq!(ltype, rtype);
                // let t = if let Some(utype) = self.unify(ltype, rtype) {
                //     utype
                // } else {
                //     self.new_type_variable()
                // };
                // let t = self.new_type_variable();
                // dbg!(&self.types[t]);
                self.set(node_id, ltype);
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
                let _ltype = self.infer(node.lhs)?;
                let _rtype = self.infer(node.rhs)?;
                // assert_eq!(ltype, rtype);
                // self.unify(ltype, rtype);
                // if ltype != rtype {
                //     if is_integer(ltype) && is_integer_literal(rtype) {
                //         self.set(node.rhs, ltype);
                //     } else if is_integer_literal(ltype) && is_integer(rtype) {
                //         self.set(node.lhs, rtype);
                //     } else {
                //         let utype = dbg!(self.unify(ltype, rtype)).unwrap();
                //         self.set(node.lhs, utype);
                //         self.set(node.rhs, utype);
                //         // return Err(Diagnostic::error()
                //         //     .with_message(format!(
                //         //         "mismatched types: left is \"{:?}\", right is \"{:?}\"",
                //         //         self.types[ltype], self.types[rtype]
                //         //     ))
                //         //     .with_labels(vec![self.tree.label(node.token)]));
                //     }
                // } else if is_integer_literal(ltype) && is_integer_literal(rtype) {
                //     self.set(node.lhs, T::I64 as TypeId);
                //     self.set(node.rhs, T::I64 as TypeId);
                // }
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
                self.narrow(ptr, self.get(node.lhs));
                // self.unify(ptr, self.get(node.lhs));
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
                let (decltype, _) = self.get_identifier_type(node_id);
                self.set(node_id, decltype);
            }
            Tag::Type => {
                let definition = self.definitions.get(&node_id).unwrap();
                match definition {
                    Definition::BuiltInType(builtin) => match builtin {
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
                            let param_t = self.infer(ni)?;
                            self.current_parameters.push(param_t);
                        }
                        let struct_t = self.get(struct_decl_id);
                        let monomorphized_t = self.monomorphize_type(struct_t);
                        self.current_parameters.truncate(current_parameter_count);
                        self.current_struct_id = 0;
                        self.set(node_id, monomorphized_t);
                    }
                    _ => {
                        // return Err(Diagnostic::error().with_message("Undefined type"))
                    }
                }
            }
            Tag::StringLiteral => self.set(node_id, T::String as TypeId),
            Tag::IntegerLiteral => {
                let token_str = self.tree.node_lexeme(node_id);
                // self.set(
                //     node_id,
                //     smallest_integer_type(token_str.parse::<i64>().unwrap()),
                // )
                let integer_literal_t =
                    self.add_type(integer_literal_type(token_str.parse::<i64>().unwrap()));
                self.set(node_id, integer_literal_t);
                // self.set_node_type(node_id, T::IntegerLiteral as TypeId)
            }
            Tag::True | Tag::False => self.set(node_id, T::Boolean as TypeId),
            Tag::FloatLiteral => self.set(node_id, T::F32 as TypeId),
            _ => {
                panic!("infer: {:?}", node.tag);
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
    /// - Argument types
    /// - Assignment types
    /// - Return types
    /// - Conditional expressions
    fn check(&mut self, node_id: NodeId) -> Result<()> {
        let node = &self.tree.node(node_id).clone();
        // println!("check: {node_id} {:?}", node.tag);
        match node.tag {
            Tag::Root | Tag::Module | Tag::Block | Tag::IfElse => {
                for i in self.tree.range(node) {
                    self.check(self.tree.node_index(i))?;
                }
            }
            Tag::Expressions => {
                let mut ts = vec![];
                for i in self.tree.range(node) {
                    let ni = self.tree.node_index(i);
                    self.check(ni)?;
                    ts.push(self.get(ni));
                }
                let t = self.add_tuple_type(ts);
                self.set(node_id, t);
            }
            // Declarations
            Tag::FunctionDecl => {
                self.check(node.lhs)?;
                self.type_definitions.insert(self.get(node.lhs), node_id);
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
                // Check initial value expression.
                let init_expr = self.tree.node_extra(node, 1);
                if init_expr != 0 {
                    self.check(init_expr)?;

                    // let annotation_id = self.tree.node_extra(node, 0);

                    let identifiers = self.tree.node(node.lhs);
                    let expressions = self.tree.node(init_expr);

                    let rtypes = self.tree.range(expressions).fold(vec![], |mut v, i| {
                        let ti = self.get(self.tree.node_index(i));
                        let ti = self.flatten_var_type(ti);
                        match &self.types[ti] {
                            Type::Tuple { fields } => v.extend(fields),
                            _ => v.push(ti),
                        }
                        v
                    });

                    // dbg!(&rtypes);

                    if self.tree.range(identifiers).len() != rtypes.len() {
                        return Err(Diagnostic::error()
                            .with_message(format!(
                                "Expected {} initial values, got {}.",
                                self.tree.range(identifiers).len(),
                                rtypes.len()
                            ))
                            .with_labels(vec![self.tree.label(node.token)]));
                    }

                    let annotation_id = self.tree.node_extra(node, 0);

                    // Concretize rhs
                    for (id_i, expr_i) in self
                        .tree
                        .range(identifiers)
                        .zip(self.tree.range(expressions))
                    {
                        // lhs
                        let id_id = self.tree.node_index(id_i);
                        let expr_id = self.tree.node_index(expr_i);

                        // dbg!(self.get(id_id), self.get(expr_id));
                        self.narrow(self.get(id_id), self.get(expr_id));

                        self.make_concrete(id_id);
                        self.make_concrete(expr_id);
                        // dbg!(self.get(id_id), self.get(expr_id));

                        // Coerce init expr into identifier type
                        // if let Some(coerced) = self.coerce(expr_id, t) {
                        //     // Concretize identifier type
                        //     let concrete = self.concretize(id, t);
                        //     self.set_node_type(id, concrete);
                        //     self.tree.indices[i as usize] = coerced;
                        // }

                        if annotation_id != 0 {
                            let annotation_t = self.get(annotation_id);
                            if annotation_t != self.get(expr_id) {
                                return Err(Diagnostic::error()
                                .with_message(format!(
                                    "mismatched types in variable declaration: expected {:?}, got {:?}",
                                    self.types[annotation_t],
                                    self.types[self.get(expr_id)],
                                ))
                                .with_labels(vec![self.tree.label(node.token)]));
                            }
                        }
                    }
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
                // lhs must match the function return types
                if node.lhs != 0 {
                    let node_id = *self.fn_id_stack.last().unwrap();
                    let fn_t = self.get(node_id);
                    let return_t = self.add_tuple_type(self.types[fn_t].returns().clone());

                    let return_ts = self.type_ids(return_t);

                    let mut is_generic = false;
                    for t in &return_ts {
                        if let Type::TypeParameter { .. } = self.types[*t] {
                            is_generic = true;
                        }
                    }

                    // Coerce return values if not generic
                    if !is_generic {
                        self.check(node.lhs)?;
                        let expressions = self.tree.node(node.lhs);

                        for (index, i) in self.tree.range(expressions).enumerate() {
                            let ni = self.tree.node_index(i);
                            if let Some(coerced) = self.coerce(ni, return_ts[index]) {
                                self.tree.indices[i as usize] = coerced;
                            } else {
                                return Err(Diagnostic::error()
                                    .with_message(format!(
                                        "mismatched types in return: expected {:?}, got {:?}",
                                        self.types[return_ts[index]],
                                        self.types[self.get(ni)]
                                    ))
                                    .with_labels(vec![self.tree.label(node.token)]));
                            }
                        }
                    }
                    self.set(node.lhs, return_t);
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
                // dbg!(&self.types[self.get(node.lhs)]);

                self.check(node.lhs)?;

                if self.get(node.lhs) == T::Void as TypeId {
                    // Module access
                    self.infer(node.rhs)?;
                    self.check(node.rhs)?;
                    self.set(node_id, self.get(node.rhs));

                    // let rtype = self.get(node.rhs);
                    // if let Some(unified_type) = self.unify(self.get(node_id), rtype) {
                    //     let concrete = self.concretize(unified_type);
                    //     self.set(node_id, concrete);
                    // }
                    return Ok(());
                }

                // Struct access

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
                    // dbg!(&ltype);
                    // dbg!(&self.types[ltype]);
                    panic!("{}.{}", self.tree.name(node.lhs), self.tree.name(node.rhs));
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
                // if let Some(unified_type) = self.unify(self.get(node_id), rtype) {
                //     let concrete = self.concretize(unified_type);
                //     self.set(node_id, concrete);
                // }
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
                if let Type::Pointer { typ, .. } = self.ty(node.lhs) {
                    // Narrow since a type variable was inferred.
                    self.narrow(self.get(node_id), *typ);
                } else {
                    return Err(Diagnostic::error()
                        .with_message(format!(
                            "type \"{}\" cannot be dereferenced",
                            self.format_type(self.get(node.lhs))
                        ))
                        .with_labels(vec![self.tree.label(self.tree.node(node.lhs).token)]));
                }
                // if let None = self.unify(self.get_t(node.lhs), Type::AnyPointer) {
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

                let (callee_t, _) = self.get_identifier_type(callee_id);

                let parameter_ts = self.types[callee_t].parameters();
                assert_eq!(parameter_ts.len(), 2);
                let lparam_t = parameter_ts[0];
                let rparam_t = parameter_ts[1];

                if let Some(coerced) = self.coerce(node.lhs, lparam_t) {
                    self.tree.nodes[node_id as usize].lhs = coerced;
                }
                if let Some(coerced) = self.coerce(node.rhs, rparam_t) {
                    self.tree.nodes[node_id as usize].rhs = coerced;
                }

                let return_t = if let Type::Function { returns, .. } = &self.types[callee_t] {
                    if !returns.is_empty() {
                        self.add_tuple_type(returns.clone())
                    } else {
                        T::Void as TypeId
                    }
                } else {
                    // Callee type is not a function (e.g. a cast).
                    panic!();
                };

                self.set(node_id, return_t);
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
                // if let Some(unified) = self.unify(self.get(node.lhs), self.get(node.rhs)) {
                //     self.set(node.lhs, unified);
                //     self.set(node.rhs, unified);
                // }
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
                self.set(node_id, self.get(node.lhs));
                // self.unify(self.get(node_id), self.get(node.lhs));
                // self.set(node_id, self.get(node.lhs));
                // if let Some(coerced) = self.coerce(node.lhs, self.get(node_id)) {
                //     self.tree.node_mut(node_id).lhs = coerced;
                // } else {
                //     panic!(
                //         "node: {}, node.lhs: {}",
                //         self.get(node_id),
                //         self.get(node.lhs)
                //     );
                // }
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
                // if let Some(unified) = self.unify(self.get(node.lhs), self.get(node.rhs)) {
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
                // }
            }
            Tag::Not => {
                self.check(node.lhs)?;
                // if let Some(coerced) = self.coerce(node.lhs, T::Boolean as TypeId) {
                //     self.tree.node_mut(node_id).lhs = coerced;
                // } else {
                //     panic!();
                // }
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
                if is_none(self.get(callee_id)) || self.tree.node(callee_id).tag == Tag::Type {
                    self.infer(callee_id)?;
                }

                // 3. Finalize callee
                self.check(callee_id)?;

                // 4. Finalize arguments.

                ////////////////////////////////////////////////////////////////

                let (callee_t, is_built_in_type) = self.get_identifier_type(callee_id);
                if is_built_in_type {
                    // panic!("{:?} {:?}, {is_built_in_type}", self.get(node_id), callee_t);
                    self.set(node_id, callee_t);
                    self.make_concrete(node_id);
                    return Ok(());
                }
                let callee_t = self.get(callee_id);

                // assert_eq!(
                //     callee_t,
                //     self.get(callee_id),
                //     "{:?} {:?}",
                //     &self.types[callee_t],
                //     &self.types[self.get(callee_id)]
                // );

                // Coerce argument types
                let parameter_ts = self.types[callee_t].parameters().clone();
                let arguments = self.tree.node(node.rhs);
                for (index, i) in self.tree.range(arguments).enumerate() {
                    let ni = self.tree.node_index(i);
                    if let Some(coerced) = self.coerce(ni, parameter_ts[index]) {
                        self.tree.indices[i as usize] = coerced;
                    }
                }
                let parameters_t = self.add_tuple_type(parameter_ts.to_vec());
                self.set(node.rhs, parameters_t);

                if let Type::Function { returns, .. } = &self.types[callee_t] {
                    let return_t = self.add_tuple_type(returns.clone());
                    self.set(node_id, return_t);
                }

                // self.set(node_id)

                // 4. Set return type.
                // let callee_type = self.get(callee_id);
                // let type_id = if let Type::Function {
                //     parameters: _,
                //     returns,
                // } = &self.types[callee_type]
                // {
                //     if !returns.is_empty() {
                //         self.add_tuple_type(returns.clone())
                //     } else {
                //         T::Void as TypeId
                //     }
                // } else {
                //     // Callee type is not a function (e.g. a cast).
                //     callee_type as TypeId
                // };
            }
            Tag::Subscript => {
                // rhs must be an i64
                self.check(node.lhs)?;
                self.check(node.rhs)?;

                let element_t = self.ty(node.lhs).element_type();
                // self.unify(self.get(node_id), element_type);
                self.narrow(self.get(node_id), element_t);

                if let Some(coerced) = self.coerce(node.rhs, T::I64 as TypeId) {
                    self.tree.node_mut(node_id).rhs = coerced;
                } else {
                    panic!();
                }
            }
            // Simple expressions
            Tag::Identifier => {}
            // Literal expressions
            Tag::IntegerLiteral => {}
            Tag::FloatLiteral => {}
            Tag::True | Tag::False => {}
            Tag::StringLiteral => {}
            Tag::Conversion => {}
            Tag::Type => {}
            _ => {
                unreachable!("Unexpected node type in check: {:?}", node.tag);
            }
        }
        self.make_concrete(node_id);
        Ok(())
    }

    fn get_identifier_type(&mut self, node_id: NodeId) -> (TypeId, bool) {
        let decl = self.definitions.get(&node_id);
        if let Some(lookup) = decl {
            match lookup {
                Definition::User(decl_id) | Definition::Resolved(decl_id) => {
                    (self.get(*decl_id), false)
                }
                Definition::BuiltInType(type_id) => {
                    if *type_id == T::Pointer {
                        (self.get(node_id), true)
                    } else {
                        (*type_id as TypeId, true)
                    }
                }
                Definition::BuiltInFunction(built_in_fn) => (
                    self.add_function_type(
                        self.types[*self.builtin_function_types.get(built_in_fn).unwrap()].clone(),
                    ),
                    false,
                ),
                Definition::Foreign(_) => (T::Void as TypeId, false),
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

    // fn coerce(&mut self, node_id: NodeId, t: TypeId) -> Option<NodeId> {
    //     let node_type = self.concretize(self.get(node_id));
    //     println!(
    //         "  - coerce({node_id}: {} -> {node_type}, {t})",
    //         self.get(node_id)
    //     );
    //     self.set(node_id, node_type);
    //     if is_none(t) || !dbg!(self.is_subtype(node_type, t)) {
    //         return None;
    //     }
    //     // node_type <: t
    //     if dbg!(self.is_subtype_generic(node_type, t))
    //         || node_type == t
    //         || matches!(self.types[node_type], Type::Pointer { .. })
    //     {
    //         // println!("{node_type} {t}");
    //         return Some(node_id);
    //     }
    //     let conversion_node_id =
    //         self.tree
    //             .add_node(Tag::Conversion, self.tree.node(node_id).token, node_id, 0);
    //     self.set(conversion_node_id, t);
    //     Some(conversion_node_id as NodeId)
    // }

    fn coerce(&mut self, node_id: NodeId, t: TypeId) -> Option<NodeId> {
        let concrete = self.make_concrete(node_id);
        // println!(
        //     "  - coerce({node_id}: {} -> {concrete}, {t})",
        //     self.get(node_id)
        // );
        if is_none(t) {
            return None;
        }
        if let Type::TypeParameter { .. } = self.ty(node_id) {
            return Some(node_id);
        }
        if concrete == self.concretize(t) {
            // No conversion necessary.
            return Some(node_id);
        }
        if !self.is_subtype(concrete, t) {
            // Cannot convert.
            return None;
        }
        // Convert.
        let conversion_node_id =
            self.tree
                .add_node(Tag::Conversion, self.tree.node(node_id).token, node_id, 0);
        self.set(conversion_node_id, t);
        Some(conversion_node_id as NodeId)
    }

    fn make_concrete(&mut self, node_id: NodeId) -> TypeId {
        let concrete = self.concretize(self.get(node_id));
        self.set(node_id, concrete);
        concrete
    }

    fn concretize(&mut self, t: TypeId) -> TypeId {
        if is_none(t) || is(t, T::Void) {
            t
        } else {
            self.fullconcrete(t)
        }
    }

    fn fullconcrete(&mut self, t: TypeId) -> TypeId {
        match self.types[t].clone() {
            Type::Numeric { .. } => {
                if is_integer_literal(t) {
                    t - 4
                } else {
                    t
                }
            }
            Type::NumericLiteral { max, .. } => max,
            Type::Void
            | Type::None
            | Type::Any
            | Type::Boolean
            | Type::Never
            | Type::Struct { .. }
            | Type::Type { .. } => t,
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
            Type::Array { typ, length, .. } => {
                let t = self.fullconcrete(typ);
                if is(t, T::Void) {
                    panic!()
                } else {
                    self.add_array_type(t, length)
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

            // _ => unreachable!("Unexpected type! {:?}", &self.types[t]),
            Type::TypeParameter { .. } => t,
        }
    }

    fn narrow(&mut self, a: TypeId, b: TypeId) -> Option<TypeId> {
        // println!("narrow(a: {a}, b:{b})");
        if is_none(a) || is_none(b) {
            return None;
        }
        if a == b {
            return Some(a);
        }
        let type_a = self.types[a].clone();
        let b_concrete = self.flatten_var_type(b);
        let type_b = self.types[b_concrete].clone();
        match (type_a, type_b) {
            (Type::NumericLiteral { min, max, .. }, Type::Numeric { .. }) => {
                if b < max && b >= min {
                    let (_, max) = self.types[a].min_max_mut();
                    *max = b
                }
            }
            (Type::Parameter { binding, .. }, _) if self.is_subtype(binding, b) => {
                // dbg!("binding <: b");
                if let Some(x) = self.narrow(binding, b_concrete) {
                    let binding = self.types[a].binding_mut();
                    // dbg!(*binding);
                    *binding = x;
                } else {
                    let binding = self.types[a].binding_mut();
                    *binding = b_concrete;
                }
            }
            (Type::Tuple { fields: aa }, Type::Tuple { fields: bb }) => {
                if aa.len() != bb.len() {
                    panic!()
                }
                aa.iter().zip(bb.iter()).for_each(|(a, b)| {
                    self.narrow(*a, *b);
                })
            }
            _ => {
                // panic!("a: {}, b: {}", a, b)
                return None;
            }
        }
        Some(a)
    }

    fn is_subtype(&self, a: TypeId, b: TypeId) -> bool {
        if is_none(a) || is_none(b) {
            return false;
        }
        if a == b {
            return true;
        }
        if is(b, T::Any) {
            return true;
        }
        match (self.types[a].clone(), self.types[b].clone()) {
            (Type::Never, _) | (Type::Any, _) => true,
            (Type::Numeric { bytes: bytes_a, .. }, Type::Numeric { bytes: bytes_b, .. }) => {
                bytes_a <= bytes_b
            }
            (Type::NumericLiteral { max, min, .. }, Type::Numeric { .. }) => b <= max && b >= min,
            (Type::Pointer { typ: a, .. }, Type::Pointer { typ: b, .. }) => self.is_subtype(a, b),
            (Type::Tuple { fields: a }, Type::Tuple { fields: b }) => a
                .iter()
                .zip(b.iter())
                .all(|(ai, bi)| self.is_subtype(*ai, *bi)),
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
                let mut fn_type_ids = vec![];
                'outer: for &definition in overload_set {
                    match definition {
                        Definition::User(fn_decl_id) => {
                            let resolution = Definition::Resolved(fn_decl_id);
                            let fn_decl = self.tree.node(fn_decl_id);
                            let fn_type_id = self.get(fn_decl.lhs);
                            // If the arguments could match the parameters
                            fn_type_ids.push(fn_type_id);
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
                            fn_type_ids.push(fn_type_id);
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
                let argument_types: Vec<&Type> =
                    argument_ids.iter().map(|id| self.ty(*id)).collect();
                return Err(Diagnostic::error()
                    .with_message(format!(
                        "failed to find matching overload for arguments: {:?}",
                        argument_types
                    ))
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
                                let arg_type = flat_argument_types[pi];
                                match *self.ty(ni) {
                                    Type::TypeParameter { index, .. } => {
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
                                    Type::Pointer { typ, .. } => {
                                        if let Type::TypeParameter { index, .. } = self.types[typ] {
                                            let param_type = type_arguments[index];
                                            if param_type == T::None as TypeId {
                                                type_arguments[index] =
                                                    self.types[arg_type].element_type();
                                                // dbg!(type_arguments[index]);
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
                                    _ => {}
                                }
                            }
                            let mut return_type_ids = vec![];
                            if returns_id != 0 {
                                assert_eq!(returns.tag, Tag::Expressions);
                                for i in self.tree.range(returns) {
                                    let ni = self.tree.node_index(i);
                                    let ti = self.get(ni);
                                    let ti = match self.types[ti] {
                                        Type::TypeParameter { index, .. } => type_arguments[index],
                                        Type::Pointer { typ, .. } => {
                                            if let Type::TypeParameter { index, .. } =
                                                self.types[typ]
                                            {
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
                                parameters: flat_argument_types.into(),
                                returns: return_type_ids,
                            });
                            self.set(callee_id, callee_type_id);
                            return Ok(());
                        }

                        // Non-parametric procedure: just type-check the arguments.
                        // let fn_t = self.get(fn_decl.lhs);
                        // dbg!(&argument_ids);
                        // dbg!(&self.types[fn_t]);
                        return Ok(());
                        // return self
                        //     .check_arguments(fn_type_id, argument_ids)
                        //     .map_err(|err| {
                        //         err.with_labels(vec![self
                        //             .tree
                        //             .label(self.tree.node(callee_id).token)])
                        //     });
                    }
                    _ => {
                        panic!()
                    }
                }
            }
            Definition::BuiltInFunction(built_in_function) => {
                let fn_t = *self.builtin_function_types.get(&built_in_function).unwrap();
                return self.check_arguments(fn_t, argument_ids).map_err(|err| {
                    err.with_labels(vec![self.tree.label(self.tree.node(callee_id).token)])
                });
            }
            Definition::Foreign(_) => {}
            Definition::BuiltInType(built_in_t) => self.set(callee_id, built_in_t as TypeId),
            _ => unreachable!("Definition not found: {}", self.tree.name(callee_id)),
        }
        Ok(())
    }

    ///  
    fn check_arguments(&mut self, fn_t: TypeId, argument_ids: &[NodeId]) -> Result<()> {
        let fn_type = &self.types[fn_t].clone();
        let parameter_ts = fn_type.parameters();
        let mut parameter_index = 0;
        for node_id in argument_ids {
            let t = self.get(*node_id);
            if parameter_index >= parameter_ts.len() {
                return Err(Diagnostic::error().with_message(format!(
                    "invalid function call: expected {:?} arguments, got {:?}",
                    parameter_ts.len(),
                    argument_ids.len()
                )));
            }
            for &arg_t in &self.type_ids(t) {
                let param_t = parameter_ts[parameter_index];
                parameter_index += 1;
                // dbg!(arg_t, param_t);
                if !self.is_subtype(arg_t, param_t) {
                    return Err(Diagnostic::error().with_message(format!(
                        "mismatched types in function call: expected {:?}, got {:?}",
                        self.types[param_t], self.types[arg_t]
                    )));
                }
            }
        }
        if parameter_index != parameter_ts.len() {
            return Err(Diagnostic::error().with_message(format!(
                "invalid function call: expected {:?} arguments, got {:?}",
                parameter_ts.len(),
                argument_ids.len()
            )));
        }
        Ok(())
    }

    fn type_ids(&self, t: TypeId) -> Vec<TypeId> {
        match &self.types[t] {
            Type::Tuple { fields } => fields.clone(),
            _ => vec![t],
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
            Type::TypeParameter { index, .. } => self.current_parameters[index],
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
            Type::TypeParameter { .. } => true,
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

    fn ty(&self, node_id: NodeId) -> &Type {
        &self.types[self.tree.node(node_id).ty]
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
            Type::Numeric {
                literal,
                floating,
                signed,
                bytes,
            } => format!(
                "{}{}{}",
                if *floating {
                    "f"
                } else if *signed {
                    "i"
                } else {
                    "u"
                },
                bytes * 8,
                if *literal { "lit" } else { "" }
            ),
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

fn is_integer_literal(t: TypeId) -> bool {
    t >= T::CI8 as TypeId && t <= T::CI64 as TypeId
}

#[inline]
fn is_none(type_id: TypeId) -> bool {
    type_id == T::None as TypeId
}

#[inline]
fn is(type_id: TypeId, ty: T) -> bool {
    type_id == ty as TypeId
}
