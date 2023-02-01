use crate::{
    analyze::{BuiltInFunction, Definition},
    parse::{Node, NodeId, Tag, Tree},
    workspace::{Result, Workspace},
};
use codespan_reporting::diagnostic::Diagnostic;
use std::{collections::HashMap, hash::Hash};

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    None,
    Never,
    Void,
    Any,
    Boolean,
    IntegerLiteral,
    String,
    Type,

    Array {
        typ: TypeId,
        length: usize,
        is_generic: bool,
    },
    Function {
        parameters: Vec<TypeId>,
        returns: Vec<TypeId>,
    },
    Pointer {
        typ: TypeId,
        is_generic: bool,
    },
    Struct {
        fields: Vec<TypeId>,
        is_generic: bool,
    },
    Tuple {
        fields: Vec<TypeId>,
    },
    Parameter {
        index: usize,
    },
    Numeric {
        floating: bool,
        signed: bool,
        bytes: u8,
    },
}

impl Type {
    pub fn parameters(&self) -> &Vec<TypeId> {
        if let Type::Function { parameters, .. } = self {
            parameters
        } else {
            unreachable!()
        }
    }
    pub fn returns(&self) -> &Vec<TypeId> {
        if let Type::Function { returns, .. } = self {
            returns
        } else {
            unreachable!()
        }
    }
    pub fn is_signed(&self) -> bool {
        match self {
            Self::Numeric { signed, .. } => *signed,
            _ => unreachable!("is_signed is only valid for integer types"),
        }
    }
}

pub type TypeId = usize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum T {
    None,
    Never,
    Void,
    Any,
    Boolean,

    I8,
    I16,
    I32,
    I64,
    CI8,
    CI16,
    CI32,
    CI64,

    U8,
    U16,
    U32,
    U64,
    CU8,
    CU16,
    CU32,
    CU64,

    F32,
    F64,

    IntegerLiteral,

    Type,
    Array,
    Pointer,

    String,

    Count,

    // Prelude types
    PointerU8,
    // String = T::Count as isize + 4,
}

enum CallResult {
    Ok,
    WrongArgumentCount(usize),
    WrongArgumentTypes(TypeId, TypeId),
}

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
    type_definitions: HashMap<TypeId, NodeId>,
    type_parameters: TypeParameters,
    current_parameters: Vec<TypeId>,
    // This will have to be updated for nested struct definitions.
    current_fn_type_id: Option<TypeId>,
    current_struct_id: NodeId,

    pub types: Vec<Type>,
    pub node_types: Vec<TypeId>,
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
        types[T::I8 as TypeId] = Type::Numeric {
            floating: false,
            signed: true,
            bytes: 1,
        };
        types[T::I16 as TypeId] = Type::Numeric {
            floating: false,
            signed: true,
            bytes: 2,
        };
        types[T::I32 as TypeId] = Type::Numeric {
            floating: false,
            signed: true,
            bytes: 4,
        };
        types[T::I64 as TypeId] = Type::Numeric {
            floating: false,
            signed: true,
            bytes: 8,
        };
        // Unsigned integers
        types[T::U8 as TypeId] = Type::Numeric {
            floating: false,
            signed: false,
            bytes: 1,
        };
        types[T::U16 as TypeId] = Type::Numeric {
            floating: false,
            signed: false,
            bytes: 2,
        };
        types[T::U32 as TypeId] = Type::Numeric {
            floating: false,
            signed: false,
            bytes: 4,
        };
        types[T::U64 as TypeId] = Type::Numeric {
            floating: false,
            signed: false,
            bytes: 8,
        };
        // Floating-point numbers
        types[T::F32 as TypeId] = Type::Numeric {
            floating: true,
            signed: true,
            bytes: 4,
        };
        types[T::F64 as TypeId] = Type::Numeric {
            floating: true,
            signed: true,
            bytes: 8,
        };
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
        let binary_int_op_type = Type::Function {
            parameters: vec![T::I64 as TypeId, T::I64 as TypeId],
            returns: vec![T::I64 as TypeId],
        };
        types.push(binary_int_op_type);
        let binary_int_op_type_id = types.len() - 1;
        types.push(Type::Function {
            parameters: vec![T::Any as TypeId],
            returns: vec![T::I64 as TypeId],
        });
        let sizeof_type_id = types.len() - 1;
        let fn_types = [
            (BuiltInFunction::Add, binary_int_op_type_id),
            (BuiltInFunction::Mul, binary_int_op_type_id),
            (BuiltInFunction::SizeOf, sizeof_type_id),
        ];
        for (tag, type_id) in fn_types {
            builtin_function_types.insert(tag, type_id);
        }
        let node_count = tree.nodes.len();
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
            types,
            node_types: vec![T::Void as TypeId; node_count],
        }
    }

    pub fn results(self) -> (Vec<Type>, Vec<TypeId>, TypeParameters) {
        (self.types, self.node_types, self.type_parameters)
    }

    ///
    pub fn check(&mut self) -> Result<()> {
        let root = &self.tree.node(0).clone();
        // Resolve root declarations.
        for i in root.lhs..root.rhs {
            let module_id = self.tree.node_index(i);
            let module = self.tree.node(module_id);
            for i in module.lhs..module.rhs {
                let decl_id = self.tree.node_index(i);
                self.infer_declaration_type(decl_id)?;
            }
        }
        println!(
            "{}",
            crate::format_red!("Done inferring module declarations")
        );
        self.infer_range(root)?;
        Ok(())
    }

    fn infer_declaration_type(&mut self, node_id: NodeId) -> Result<()> {
        if node_id == 0 || self.node_types[node_id as usize] != T::Void as TypeId {
            return Ok(());
        }
        let node = self.tree.node(node_id);
        match node.tag {
            Tag::FunctionDecl => {
                // prototype
                let type_id = self.infer_node(node.lhs)?;
                self.type_definitions.insert(type_id, node_id);
            }
            Tag::Struct => {
                let type_id = self.infer_node(node_id)?;
                self.type_definitions.insert(type_id, node_id);
            }
            _ => {}
        };
        Ok(())
    }

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
        let current_type_id = self.node_types[node_id as usize];
        if current_type_id != T::Void as TypeId {
            return Ok(current_type_id);
        }
        let mut inferred_node_ids = vec![];
        let mut inferred_type_ids = vec![];
        let mut result: TypeId = match node.tag {
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
                let definition = *self.definitions.get(&callee_id).unwrap_or_else(|| {
                    panic!("Definition not found: {}", self.tree.name(callee_id))
                });

                let argument_types = vec![
                    (
                        node.lhs,
                        self.infer_node_with_type(node.lhs, parent_type_id)?,
                    ),
                    (
                        node.rhs,
                        self.infer_node_with_type(node.rhs, parent_type_id)?,
                    ),
                ];

                self.infer_call(callee_id, &definition, argument_types)?;

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
            Tag::Div
            | Tag::Sub
            | Tag::BitwiseAnd
            | Tag::BitwiseOr
            | Tag::BitwiseShiftL
            | Tag::BitwiseShiftR
            | Tag::BitwiseXor => self.infer_binary_node(node)?,
            Tag::Assign => {
                // if self.tree.node(node.lhs).tag == Tag::Identifier {
                //     let decl_identifier_id = self
                //         .definitions
                //         .get_definition_id(node.lhs, "cannot get lvalue definition");
                //     let decl_node_id = self.tree.node(decl_identifier_id).lhs;
                //     dbg!(decl_node_id);
                //     dbg!(self.tree.is_decl_const(decl_node_id));
                //     if self.tree.is_decl_const(decl_node_id) {
                //         return Err(Diagnostic::error()
                //             .with_message(format!(
                //                 "cannot assign to constant \"{}\"",
                //                 self.tree.node_lexeme(node.lhs)
                //             ))
                //             .with_labels(vec![self.tree.label(node.token)]));
                //     }
                // }
                let ltype = self.infer_node(node.lhs)?;
                let rtype = self.infer_node_with_type(node.rhs, Some(&ltype))?;
                // if is_integer(ltype) && rtype == T::IntegerLiteral as TypeId {
                //     self.node_types[node.rhs as usize] = ltype;
                // } else

                // let lvalues = self.tree.node(node.lhs);
                // let mut ltypes = Vec::<TypeId>::new();
                // for i in self.tree.range(lvalues) {
                //     let ni = self.tree.node_index(i);
                //     let ti = &self.node_types[ni as usize];
                //     match ti {
                //         Single(t) => ltypes.push(*t),
                //         Multiple(ts) => ltypes.extend(ts),
                //     }
                // }

                // let rvalues = self.tree.node(node.rhs);
                // let mut rtypes = Vec::<TypeId>::new();
                // for i in self.tree.range(rvalues) {
                //     let ni = self.tree.node_index(i);
                //     // let ti = &self.node_types[ni as usize];
                //     let ti = self.infer_node_with_type(ni, parent_type_id)?;
                //     match ti {
                //         Single(t) => rtypes.push(t),
                //         Multiple(ts) => rtypes.extend(ts),
                //     }
                // }

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
            Tag::Block | Tag::Expressions | Tag::IfElse | Tag::Module | Tag::Parameters => {
                self.infer_range(node)?
            }
            Tag::Call => {
                // Argument expressions
                self.infer_node(node.rhs)?;
                let expressions = self.tree.rchild(node);

                // Callee
                let callee_id = node.lhs;
                let definition = *self.definitions.get(&callee_id).unwrap_or_else(|| {
                    panic!("Definition not found: {}", self.tree.name(callee_id))
                });

                let argument_types: Vec<(NodeId, TypeId)> = self
                    .tree
                    .range(expressions)
                    .map(|i| {
                        let ni = self.tree.node_index(i);
                        (ni, self.node_types[ni as usize])
                    })
                    .collect();

                self.infer_call(callee_id, &definition, argument_types)?;

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
                self.set_node_type(node_id, fn_type);
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
                        self.node_types[node.rhs as usize] = ltype;
                        return Ok(ltype);
                    } else if ltype == T::IntegerLiteral as TypeId && is_integer(rtype) {
                        self.node_types[node.lhs as usize] = rtype;
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
                    self.node_types[node.lhs as usize] = T::I64 as TypeId;
                    self.node_types[node.rhs as usize] = T::I64 as TypeId;
                }
                T::Boolean as TypeId
            }
            Tag::Not => self.infer_node(node.lhs)?,
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
            Tag::IntegerLiteral => match parent_type_id {
                Some(type_id) if is_integer(*type_id) => *type_id,
                _ => T::IntegerLiteral as TypeId,
            },
            Tag::FloatLiteral => T::F32 as TypeId,
            Tag::True | Tag::False => T::Boolean as TypeId,
            Tag::Negation => {
                if let Some(type_id) = parent_type_id {
                    self.infer_node_with_type(node.lhs, Some(type_id))?
                } else {
                    self.infer_node_with_type(node.lhs, Some(&(T::I64 as TypeId)))?
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
                        });
                        self.set_node_type(node_id, self.types.len() - 1);
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
                    let ni = self.tree.node_index(i) as usize;
                    parameters.push(self.node_types[ni]);
                }
                if rets.tag == Tag::Expressions {
                    for i in rets.lhs..rets.rhs {
                        let ni = self.tree.node_index(i) as usize;
                        returns.push(self.node_types[ni]);
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
                        });
                        self.set_node_type(node_id, param_type);
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
                    let ni = self.tree.node_index(i) as usize;
                    fields.push(self.node_types[ni]);
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
                        let struct_type_id = self.node_types[struct_decl_id as usize];
                        let specified_type = self.monomorphize_type(struct_type_id);
                        self.current_parameters.truncate(current_parameter_count);
                        self.current_struct_id = 0;
                        specified_type
                    }
                    _ => return Err(Diagnostic::error().with_message("Undefined type")),
                }
            }
            Tag::VariableDecl => {
                // lhs: type-expr
                // rhs: init-expr
                let identifiers = &self.tree.node(node.lhs).clone();
                let annotation_id = self.tree.node_extra(node, 0);
                let rvalues_id = self.tree.node_extra(node, 1);

                let annotation = self.infer_node(annotation_id)?;
                if rvalues_id == 0 {
                    // Set lhs types.
                    for i in self.tree.range(identifiers) {
                        let ni = self.tree.node_index(i);
                        self.set_node_type(ni, annotation);
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
                        let ti = &self.node_types[ni as usize];
                        match &self.types[*ti] {
                            Type::Tuple { fields } => rtypes.extend(fields),
                            _ => rtypes.push(*ti),
                        }
                        if *ti == T::IntegerLiteral as TypeId {
                            let inferred_type = infer_type(annotation, *ti)?;
                            self.node_types[ni as usize] = inferred_type;
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
                        let inferred_type = infer_type(annotation, *ti)?;
                        self.set_node_type(ni, inferred_type);
                    }

                    let tuple_type = self.add_tuple_type(rtypes);
                    self.set_node_type(rvalues_id, tuple_type);
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

        // Set the types for any deferred nodes.
        for (ni, ti) in inferred_node_ids.iter().zip(inferred_type_ids.iter()) {
            self.set_node_type(*ni, *ti);
        }

        // After recursively calling inferring the types of child nodes, set the type of the parent.
        self.set_node_type(node_id, result);
        Ok(result)
    }

    ///
    fn infer_binary_node(&mut self, node: &Node) -> Result<TypeId> {
        let ltype = self.infer_node(node.lhs)?;
        let rtype = self.infer_node(node.rhs)?;
        if ltype != rtype {
            if is_integer(ltype) && rtype == T::IntegerLiteral as TypeId {
                self.node_types[node.rhs as usize] = ltype;
                return Ok(ltype);
            } else if ltype == T::IntegerLiteral as TypeId && is_integer(rtype) {
                self.node_types[node.lhs as usize] = rtype;
                return Ok(rtype);
            }

            return Err(Diagnostic::error()
                .with_message(format!(
                    "mismatched types: left is \"{:?}\", right is \"{:?}\"",
                    self.types[ltype], self.types[rtype]
                ))
                .with_labels(vec![self.tree.label(node.token)]));
        }
        Ok(ltype)
    }

    fn infer_call(
        &mut self,
        callee_id: NodeId,
        definition: &Definition,
        argument_types: Vec<(NodeId, TypeId)>,
    ) -> Result<()> {
        match definition {
            Definition::Overload(overload_set_id) => {
                let overload_set = self.overload_sets.get(overload_set_id).unwrap();

                'outer: for &definition in overload_set {
                    match definition {
                        Definition::User(fn_decl_id) => {
                            let resolution = Definition::Resolved(fn_decl_id);
                            let fn_decl = self.tree.node(fn_decl_id);
                            let fn_type_id = self.node_types[fn_decl.lhs as usize];
                            if let CallResult::Ok =
                                self.check_arguments(&argument_types, fn_type_id)
                            {
                                self.definitions.insert(callee_id, resolution);
                                return Ok(());
                            } else {
                                continue 'outer;
                            }
                        }
                        Definition::BuiltInFunction(built_in_function) => {
                            let fn_type_id =
                                *self.builtin_function_types.get(&built_in_function).unwrap();
                            if let CallResult::Ok =
                                self.check_arguments(&argument_types, fn_type_id)
                            {
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
                let fn_decl = self.tree.node(*definition_id);
                let prototype = &self.tree.node(fn_decl.lhs).clone();
                match prototype.tag {
                    Tag::Prototype => {
                        if prototype.lhs != 0 {
                            let mut flat_argument_types = vec![];
                            for (node_id, arg_type_ids) in argument_types {
                                if let Type::Tuple { fields } = &self.types[arg_type_ids] {
                                    flat_argument_types.extend(fields);
                                } else {
                                    if arg_type_ids == T::IntegerLiteral as TypeId {
                                        self.set_node_type(node_id, T::I64 as TypeId);
                                        flat_argument_types.push(T::I64 as TypeId);
                                        continue;
                                    }
                                    flat_argument_types.push(arg_type_ids);
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
                                let ti = self.node_types[ni as usize];
                                let arg_type = flat_argument_types[pi];
                                if let Type::Parameter { index } = self.types[ti] {
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
                                    let ti = self.node_types[ni as usize];
                                    let ti = match self.types[ti] {
                                        Type::Parameter { index } => type_arguments[index],
                                        Type::Pointer { typ, .. } => {
                                            if let Type::Parameter { index } = self.types[typ] {
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
                            if let Err(mut occupied_err) = self.type_parameters.try_insert(
                                *definition_id,
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
                            self.set_node_type(callee_id, callee_type_id);
                            return Ok(());
                        }

                        // Non-parametric procedure: just type-check the arguments.
                        let fn_type_id = self.node_types[fn_decl.lhs as usize];
                        match self.check_arguments(&argument_types, fn_type_id) {
                            CallResult::WrongArgumentCount(parameter_count) => {
                                return Err(Diagnostic::error()
                                    .with_message(format!(
                                        "invalid function call: expected {:?} arguments, got {:?}",
                                        parameter_count,
                                        argument_types.len()
                                    ))
                                    .with_labels(vec![self
                                        .tree
                                        .label(self.tree.node(callee_id).token)]));
                            }
                            CallResult::WrongArgumentTypes(param_type_id, arg_type_id) => {
                                return Err(Diagnostic::error().with_message(format!(
                                    "mismatched types in function call: expected {:?} argument, got {:?}",
                                    self.types[param_type_id],
                                    self.types[arg_type_id])
                                ).with_labels(vec![self
                                    .tree
                                    .label(self.tree.node(callee_id).token)]));
                            }
                            _ => {}
                        }
                    }
                    _ => {
                        panic!()
                    }
                }
            }
            Definition::BuiltInFunction(built_in_function) => {
                let fn_type_id = *self.builtin_function_types.get(built_in_function).unwrap();
                match self.check_arguments(&argument_types, fn_type_id) {
                    CallResult::WrongArgumentCount(parameter_count) => {
                        return Err(Diagnostic::error()
                            .with_message(format!(
                                "invalid function call: expected {:?} arguments, got {:?}",
                                parameter_count,
                                argument_types.len()
                            ))
                            .with_labels(vec![self.tree.label(self.tree.node(callee_id).token)]));
                    }
                    CallResult::WrongArgumentTypes(param_type_id, arg_type_id) => {
                        return Err(Diagnostic::error().with_message(format!(
                            "mismatched types in built-in function call: expected {:?} argument, got {:?}",
                            self.types[param_type_id],
                            self.types[arg_type_id])
                        ).with_labels(vec![self
                            .tree
                            .label(self.tree.node(callee_id).token)]));
                    }
                    _ => {}
                }
            }
            Definition::Foreign(_) => {}
            Definition::BuiltIn(built_in_type) => {
                self.set_node_type(callee_id, *built_in_type as TypeId)
            }
            _ => unreachable!("Definition not found: {}", self.tree.name(callee_id)),
        }
        Ok(())
    }

    fn check_arguments(
        &mut self,
        argument_types: &Vec<(NodeId, TypeId)>,
        fn_type_id: TypeId,
    ) -> CallResult {
        let fn_type = &self.types[fn_type_id];
        let parameter_types = fn_type.parameters();
        let mut untyped_arguments = vec![];
        let mut parameter_index = 0;
        for (node_id, arg_type_ids) in argument_types {
            if parameter_index >= parameter_types.len() {
                return CallResult::WrongArgumentCount(parameter_types.len());
            }
            for &arg_type in &self.type_ids(*arg_type_ids) {
                let param_type = parameter_types[parameter_index];
                parameter_index += 1;
                if arg_type != param_type && param_type != T::Any as TypeId {
                    // Check if untyped argument is compatible with parameter.
                    if is_integer(param_type) && arg_type == T::IntegerLiteral as TypeId {
                        untyped_arguments.push((node_id, param_type));
                        continue;
                    }
                    return CallResult::WrongArgumentTypes(param_type, arg_type);
                }
            }
        }
        if parameter_index != parameter_types.len() {
            return CallResult::WrongArgumentCount(parameter_types.len());
        }
        for (node_id, type_id) in untyped_arguments {
            self.node_types[*node_id as usize] = type_id;
        }
        CallResult::Ok
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
        match self
            .tuple_types
            .try_insert(field_types.clone(), self.types.len())
        {
            Ok(_) => self.add_type(Type::Tuple {
                fields: field_types,
            }),
            Err(err) => *err.entry.get(),
        }
    }

    /// Creates a type based on the provided type_id, with all type parameters replaced by concrete types.
    fn monomorphize_type(&mut self, type_id: TypeId) -> TypeId {
        let typ = self.types[type_id].clone();
        match typ {
            Type::Parameter { index } => self.current_parameters[index],
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

    ///
    fn set_node_type(&mut self, index: u32, type_id: TypeId) {
        self.node_types[index as usize] = type_id;
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
        // for i in 1..self.tree.nodes.len() {
        //     println!(
        //         "[{i}] {:?} {:?}: {:?}",
        //         self.tree.name(i as u32),
        //         self.tree.node(i as u32).tag,
        //         self.types[self.node_types[i] as usize]
        //     );
        // }
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

fn is_integer(type_id: TypeId) -> bool {
    type_id >= T::I8 as TypeId && type_id <= T::I64 as TypeId
        || type_id >= T::U8 as TypeId && type_id <= T::U64 as TypeId
}

fn infer_type(annotation_type: TypeId, value_type: TypeId) -> Result<TypeId> {
    let void = T::Void as TypeId;
    if annotation_type != void && (value_type == void || value_type == T::IntegerLiteral as TypeId)
    {
        // Explicit type based on annotation.
        Ok(annotation_type)
    } else if annotation_type == void && value_type == T::IntegerLiteral as TypeId {
        Ok(T::I64 as TypeId)
    } else if annotation_type == void && value_type != void {
        // Infer type based on rvalue.
        Ok(value_type)
    } else if annotation_type != value_type {
        Err(Diagnostic::error().with_message("annotation type doesn't match"))
    } else {
        Ok(annotation_type)
    }
}
