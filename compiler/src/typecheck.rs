use crate::{
    analyze::{BuiltInFunction, Definition},
    parse::{Node, NodeId, Tag, Tree},
    workspace::{Result, Workspace},
};
use codespan_reporting::diagnostic::Diagnostic;
use smallvec::SmallVec;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};
use thiserror::Error;

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

#[derive(Debug, Error)]
#[error("{0}")]
pub struct TypeError(String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Void,
    Boolean,
    Integer,
    Unsigned8,
    IntegerLiteral,
    Float,
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
    Parameter {
        index: usize,
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
            Self::Integer => true,
            Self::Unsigned8 => false,
            _ => unreachable!("is_signed is only valid for integer types"),
        }
    }
}

pub type TypeId = usize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeIds {
    Single(TypeId),
    Multiple(Vec<TypeId>),
}

use TypeIds::*;

impl TypeIds {
    pub fn first(&self) -> TypeId {
        match self {
            Single(t) => *t,
            Multiple(ts) => ts[0],
        }
    }
    pub fn all(&self) -> Vec<TypeId> {
        match self {
            Single(t) => vec![*t],
            Multiple(ts) => ts.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BuiltInType {
    Void,
    Boolean,

    Integer,
    Unsigned8,

    IntegerLiteral,
    Float,
    Type,
    Array,
    Pointer,

    Count,

    // Prelude types
    String = 11,
}

enum CallResult {
    Ok,
    WrongArgumentCount(usize),
    WrongArgumentTypes(TypeId, TypeId),
}

type TypeMap<P = Vec<TypeId>> = HashMap<P, TypeId>;
type TypeParameters = HashMap<NodeId, HashSet<Vec<TypeId>>>;

pub struct Typechecker<'a> {
    workspace: &'a mut Workspace,
    tree: &'a Tree,
    definitions: &'a mut HashMap<NodeId, Definition>,
    overload_sets: &'a HashMap<NodeId, Vec<Definition>>,

    array_types: TypeMap<(TypeId, usize)>,
    builtin_function_types: TypeMap<BuiltInFunction>,
    function_types: TypeMap<Type>,
    pointer_types: TypeMap<TypeId>,
    polymorphic_types: HashMap<NodeId, TypeMap>,
    type_definitions: HashMap<TypeId, NodeId>,
    type_parameters: TypeParameters,
    current_parameters: Vec<TypeId>,
    // This will have to be updated for nested struct definitions.
    current_fn_type_id: Option<TypeId>,
    current_struct_id: NodeId,

    pub types: Vec<Type>,
    pub node_types: Vec<TypeIds>,
}

impl<'a> Typechecker<'a> {
    pub fn new(
        workspace: &'a mut Workspace,
        tree: &'a Tree,
        definitions: &'a mut HashMap<u32, Definition>,
        overload_sets: &'a HashMap<NodeId, Vec<Definition>>,
    ) -> Self {
        let mut types = vec![Type::Void; BuiltInType::Count as usize];
        types[BuiltInType::Void as TypeId] = Type::Void;
        types[BuiltInType::Boolean as TypeId] = Type::Boolean;
        types[BuiltInType::Integer as TypeId] = Type::Integer;
        types[BuiltInType::Unsigned8 as TypeId] = Type::Unsigned8;
        types[BuiltInType::Float as TypeId] = Type::Float;
        types[BuiltInType::Type as TypeId] = Type::Type;

        // Set up string type.
        // types.push(Type::Pointer {
        //     typ: BuiltInType::Unsigned8 as TypeId,
        //     is_generic: false,
        // });
        // let ptr_u8_type = types.len() - 1;

        let pointer_types = HashMap::new();
        // let pointer_types = HashMap::from([(BuiltInType::Unsigned8 as TypeId, ptr_u8_type)]);

        // types[BuiltInType::String as TypeId] = Type::Struct {
        //     fields: vec![ptr_u8_type, BuiltInType::Integer as TypeId],
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
            parameters: vec![
                BuiltInType::Integer as TypeId,
                BuiltInType::Integer as TypeId,
            ],
            returns: vec![BuiltInType::Integer as TypeId],
        };
        types.push(binary_int_op_type);
        let binary_int_op_type_id = types.len() - 1;
        let fn_types = [
            (BuiltInFunction::Add, binary_int_op_type_id),
            (BuiltInFunction::Mul, binary_int_op_type_id),
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
            type_definitions: HashMap::new(),
            type_parameters: HashMap::new(),
            current_parameters: vec![],
            current_fn_type_id: None,
            current_struct_id: 0,
            types,
            node_types: vec![TypeIds::Single(BuiltInType::Void as TypeId); tree.nodes.len()],
        }
    }

    pub fn results(self) -> (Vec<Type>, Vec<TypeIds>, TypeParameters) {
        (self.types, self.node_types, self.type_parameters)
    }

    ///
    pub fn check(&mut self) -> Result<()> {
        let root = &self.tree.node(0);
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
        if node_id == 0 || self.node_types[node_id as usize].first() != BuiltInType::Void as TypeId
        {
            return Ok(());
        }
        let node = self.tree.node(node_id);
        match node.tag {
            Tag::FunctionDecl => {
                // prototype
                let type_id = self.infer_node(node.lhs)?.first();
                self.type_definitions.insert(type_id, node_id);
            }
            Tag::Struct => {
                let type_id = self.infer_node(node_id)?.first();
                self.type_definitions.insert(type_id, node_id);
            }
            _ => {}
        };
        Ok(())
    }

    ///
    fn infer_range(&mut self, node: &Node) -> Result<TypeIds> {
        self.infer_range_with_types(node, None)
    }

    ///
    fn infer_range_with_types(
        &mut self,
        node: &Node,
        parent_type_ids: Option<Vec<TypeIds>>,
    ) -> Result<TypeIds> {
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
                }
            }
        } else {
            for i in node.lhs..node.rhs {
                let result = self.infer_node(self.tree.node_index(i));
                if let Err(diagnostic) = result {
                    self.workspace.diagnostics.push(diagnostic);
                }
            }
        }
        Ok(Single(BuiltInType::Void as TypeId))
    }

    fn infer_node(&mut self, node_id: NodeId) -> Result<TypeIds> {
        self.infer_node_with_type(node_id, None)
    }

    ///
    fn infer_node_with_type(
        &mut self,
        node_id: NodeId,
        parent_type_id: Option<&TypeIds>,
    ) -> Result<TypeIds> {
        if node_id == 0 {
            return Ok(Single(BuiltInType::Void as TypeId));
        }
        let node = self.tree.node(node_id);
        // println!("[{}] - {:?}", node_id, node.tag);
        let current_type_id = self.node_types[node_id as usize].first();
        if current_type_id != BuiltInType::Void as TypeId {
            return Ok(Single(current_type_id));
        }
        let mut inferred_node_ids = vec![];
        let mut inferred_type_ids = vec![];
        let result: TypeIds = match node.tag {
            Tag::Access => {
                let ltype = self.infer_node(node.lhs)?.first();
                // Module access.
                if ltype == BuiltInType::Void as TypeId {
                    return self.infer_node(node.rhs);
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
                            let field_index = self.tree.node_index(field.rhs + 1);
                            self.definitions
                                .insert(node.rhs, Definition::User(field_index));
                            break;
                        }
                    }
                }

                // Struct access.
                if let (Type::Struct { fields, .. }, Definition::User(field_index)) = (
                    &self.types[ltype as usize],
                    self.definitions
                        .get(&node.rhs)
                        .expect("field index not defined"),
                ) {
                    Single(fields[*field_index as usize])
                } else {
                    return Err(Diagnostic::error().with_message("cannot access non-struct"));
                }
            }
            Tag::Address => {
                let value_type = self.infer_node(node.lhs)?.first();
                Single(self.add_pointer_type(value_type))
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
                        self.infer_node(decl_id)?.first()
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
                        Multiple(returns.clone())
                    } else {
                        Single(BuiltInType::Void as TypeId)
                    }
                } else {
                    Single(BuiltInType::Void as TypeId)
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
                let ltype = self.infer_node(node.lhs)?;
                let rtype = self.infer_node_with_type(node.rhs, Some(&ltype))?.first();
                // if is_integer(ltype) && rtype == BuiltInType::IntegerLiteral as TypeId {
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

                if node.rhs != 0 && ltype.first() != rtype {
                    return Err(Diagnostic::error()
                        .with_message(format!(
                            "mismatched types in assignment: expected {:?}, got {:?}",
                            self.types[ltype.first()],
                            self.types[rtype]
                        ))
                        .with_labels(vec![self.tree.label(node.token)]));
                }
                Single(BuiltInType::Void as TypeId)
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

                let argument_types: Vec<(NodeId, TypeIds)> = self
                    .tree
                    .range(expressions)
                    .map(|i| {
                        let ni = self.tree.node_index(i);
                        (ni, self.node_types[ni as usize].clone())
                    })
                    .collect();

                self.infer_call(callee_id, &definition, argument_types)?;

                let ltype = self.infer_node(callee_id)?.first();

                if let Type::Function {
                    parameters: _,
                    returns,
                } = &self.types[ltype]
                {
                    if !returns.is_empty() {
                        Multiple(returns.clone())
                    } else {
                        Single(BuiltInType::Void as TypeId)
                    }
                } else {
                    Single(BuiltInType::Void as TypeId)
                }
            }
            Tag::Dereference => {
                let pointer_type = self.infer_node(node.lhs)?.first();
                if let Type::Pointer { typ, .. } = self.types[pointer_type] {
                    Single(typ)
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
                let fn_type_id = fn_type.first();
                // Immediately set the declaration's type to handle recursion.
                self.set_node_type(node_id, fn_type.clone());
                // Body
                if self.current_fn_type_id.is_none() {
                    self.current_fn_type_id = Some(fn_type_id);
                }
                self.infer_node(node.rhs)?;
                fn_type
            }
            Tag::Equality | Tag::Greater | Tag::Inequality | Tag::Less => {
                self.infer_binary_node(node)?;
                Single(BuiltInType::Boolean as TypeId)
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
                        Definition::BuiltIn(type_id) => Single(*type_id as TypeId),
                        Definition::BuiltInFunction(built_in_fn) => Single(
                            self.add_function_type(
                                self.types[*self.builtin_function_types.get(built_in_fn).unwrap()]
                                    .clone(),
                            ),
                        ),
                        Definition::Overload(_) => unreachable!(),
                        // Definition::Foreign(_) => {
                        //     let ptr_type = self.add_pointer_type(2);
                        //     self.add_function_type(Type::Function {
                        //         parameters: vec![],
                        //         returns: vec![ptr_type],
                        //     })
                        // }
                        _ => Single(BuiltInType::Void as TypeId),
                    }
                } else {
                    Single(BuiltInType::Void as TypeId)
                }
            }
            Tag::IntegerLiteral => match parent_type_id {
                Some(type_id) if is_integer(type_id.first()) => Single(type_id.first()),
                _ => Single(BuiltInType::IntegerLiteral as TypeId),
            },
            Tag::True | Tag::False => Single(BuiltInType::Boolean as TypeId),
            Tag::Negation => self.infer_node(node.lhs)?,
            Tag::Prototype => {
                let mut parameters = Vec::new();
                let mut returns = Vec::new();

                let type_parameters_id = node.lhs;
                if type_parameters_id != 0 {
                    // Type parameters
                    let type_parameters = self.tree.node(type_parameters_id);
                    for i in type_parameters.lhs..type_parameters.rhs {
                        let node_id = self.tree.node_index(i);
                        self.add_type(Type::Parameter {
                            index: (i - type_parameters.lhs) as usize,
                        });
                        self.set_node_type(node_id, Single(self.types.len() - 1));
                    }
                }

                let parameters_id = self.tree.node_extra(node, 0);
                let params = self.tree.node(parameters_id);
                self.infer_node(parameters_id)?; // parameters

                let returns_id = self.tree.node_extra(node, 1);
                let rets = self.tree.node(returns_id);
                let return_type = self.infer_node(returns_id)?.first(); // returns

                assert_eq!(params.tag, Tag::Parameters);
                for i in params.lhs..params.rhs {
                    let ni = self.tree.node_index(i) as usize;
                    parameters.push(self.node_types[ni].first());
                }
                if rets.tag == Tag::Expressions {
                    for i in rets.lhs..rets.rhs {
                        let ni = self.tree.node_index(i) as usize;
                        returns.push(self.node_types[ni].first());
                    }
                } else if rets.tag == Tag::Identifier
                    || rets.tag == Tag::Type && return_type != BuiltInType::Void as TypeId
                {
                    returns.push(return_type);
                }
                Single(self.add_function_type(Type::Function {
                    parameters,
                    returns,
                }))
            }
            Tag::Return => {
                let fn_type = &self.types[self.current_fn_type_id.unwrap()];
                let return_type_ids = fn_type.returns().iter().map(|ti| Single(*ti)).collect();
                self.infer_range_with_types(node, Some(return_type_ids))?
            }
            Tag::Struct => {
                if node.lhs != 0 {
                    let type_parameters = self.tree.node(node.lhs);
                    for i in type_parameters.lhs..type_parameters.rhs {
                        let node_id = self.tree.node_index(i);
                        let param_type = self.add_type(Type::Parameter {
                            index: (i - type_parameters.lhs) as usize,
                        });
                        self.set_node_type(node_id, Single(param_type));
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
                    fields.push(self.node_types[ni].first());
                }
                let is_generic = fields.iter().any(|&type_id| self.is_generic(type_id));
                let type_id = self.add_type(Type::Struct { fields, is_generic });
                self.type_definitions.insert(type_id, node_id);
                Single(type_id)
            }
            Tag::StringLiteral => Single(BuiltInType::String as TypeId),
            Tag::Subscript => {
                let array_type = self.infer_node(node.lhs)?.first();
                self.infer_node_with_type(node.rhs, Some(&Single(BuiltInType::Integer as TypeId)))?;
                if let Type::Array { typ, .. } = self.types[array_type] {
                    Single(typ)
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
                        BuiltInType::Array => {
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
                            let value_type = self.infer_node(ni)?.first();
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
                            Single(self.add_array_type(value_type, length as usize))
                        }
                        BuiltInType::Pointer => {
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
                            let value_type = self.infer_node(ni)?.first();
                            Single(self.add_pointer_type(value_type))
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
                            let param_type = self.infer_node(ni)?.first();
                            self.current_parameters.push(param_type);
                        }
                        let struct_type_id = self.node_types[struct_decl_id as usize].first();
                        let specified_type = self.monomorphize_type(struct_type_id);
                        self.current_parameters.truncate(current_parameter_count);
                        self.current_struct_id = 0;
                        Single(specified_type)
                    }
                    _ => return Err(Diagnostic::error().with_message("Undefined type")),
                }
            }
            Tag::VariableDecl => {
                // lhs: type-expr
                // rhs: init-expr
                let identifiers = self.tree.node(node.lhs);
                let annotation_id = self.tree.node_extra(node, 0);
                let rvalues_id = self.tree.node_extra(node, 1);

                let annotation = self.infer_node(annotation_id)?.first();
                let annotation_ref = &Single(annotation);
                self.infer_node_with_type(
                    rvalues_id,
                    if annotation == BuiltInType::Void as TypeId {
                        None
                    } else {
                        Some(annotation_ref)
                    },
                )?;

                let mut r_ids = SmallVec::<[NodeId; 8]>::new();
                let mut rtypes = Vec::<TypeId>::new();
                let rvalues = self.tree.node(rvalues_id);
                for i in self.tree.range(rvalues) {
                    let ni = self.tree.node_index(i);
                    let ti = &self.node_types[ni as usize];
                    r_ids.push(ni);
                    match ti {
                        Single(t) => rtypes.push(*t),
                        Multiple(ts) => rtypes.extend(ts),
                    }
                }

                for (index, i) in self.tree.range(identifiers).enumerate() {
                    let ni = self.tree.node_index(i);
                    inferred_node_ids.push(ni);
                    let rtype = rtypes[index];
                    let inferred_type = infer_type(annotation, rtype)?;
                    if rtype == BuiltInType::IntegerLiteral as TypeId {
                        self.node_types[r_ids[index] as usize] = inferred_type.clone();
                    }
                    inferred_type_ids.push(inferred_type);
                }
                self.set_node_type(rvalues_id, Multiple(rtypes));

                Single(BuiltInType::Void as TypeId)
                // infer_type(annotation, rtypes[0])?
            }
            _ => {
                self.infer_node(node.lhs)?;
                self.infer_node(node.rhs)?;
                Single(BuiltInType::Void as TypeId)
            }
        };

        // Set the types for any deferred nodes.
        for (ni, ti) in inferred_node_ids.iter().zip(inferred_type_ids.iter()) {
            self.set_node_type(*ni, ti.clone());
        }

        // After recursively calling inferring the types of child nodes, set the type of the parent.
        self.set_node_type(node_id, result.clone());
        Ok(result)
    }

    ///
    fn infer_binary_node(&mut self, node: &Node) -> Result<TypeIds> {
        let ltype = self.infer_node(node.lhs)?.first();
        let rtype = self.infer_node(node.rhs)?.first();
        if ltype != rtype {
            if is_integer(ltype) && rtype == BuiltInType::IntegerLiteral as TypeId {
                self.node_types[node.rhs as usize] = Single(ltype);
                return Ok(Single(ltype));
            } else if ltype == BuiltInType::IntegerLiteral as TypeId && is_integer(rtype) {
                self.node_types[node.lhs as usize] = Single(rtype);
                return Ok(Single(rtype));
            }

            return Err(Diagnostic::error()
                .with_message(format!(
                    "mismatched types: left is \"{:?}\", right is \"{:?}\"",
                    self.types[ltype], self.types[rtype]
                ))
                .with_labels(vec![self.tree.label(node.token)]));
        }
        Ok(Single(ltype))
    }

    fn infer_call(
        &mut self,
        callee_id: NodeId,
        definition: &Definition,
        argument_types: Vec<(NodeId, TypeIds)>,
    ) -> Result<()> {
        match definition {
            Definition::Overload(overload_set_id) => {
                let overload_set = self.overload_sets.get(overload_set_id).unwrap();

                'outer: for &definition in overload_set {
                    match definition {
                        Definition::User(fn_decl_id) => {
                            let resolution = Definition::Resolved(fn_decl_id);
                            let fn_decl = self.tree.node(fn_decl_id);
                            let fn_type_id = self.node_types[fn_decl.lhs as usize].first();
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
                let prototype = self.tree.node(fn_decl.lhs);
                match prototype.tag {
                    Tag::Prototype => {
                        if prototype.lhs != 0 {
                            // let type_parameters = self.tree.lchild(prototype);
                            // let inner_prototype = self.tree.rchild(prototype);
                            // let parameters = self.tree.lchild(inner_prototype);
                            // let returns = self.tree.rchild(inner_prototype);
                            // let mut arg_types = Vec::new();
                            // // let mut ret_types = Vec::new();
                            // for (i, it) in (parameters.lhs..parameters.rhs).enumerate() {
                            //     let arg_id = self.tree.node_index(i as u32 + expressions.lhs);
                            //     let arg_type = self.node_types[arg_id as usize];
                            //     arg_types.push(arg_type);
                            // }
                            // // for (i, it) in (returns.lhs..returns.rhs).enumerate() {
                            // //     let ret_id = self.tree.node_index(i as u32 + expressions.lhs);
                            // //     let t = self.node_types[arg_id as usize];
                            // //     ret_types.push(t);
                            // // }
                            // self.types.push(Type::Function {
                            //     parameters: arg_types.clone(),
                            //     returns: arg_types.clone(),
                            // });
                            // if let Err(mut error) = self
                            //     .type_parameters
                            //     .try_insert(definition_id, HashSet::from([arg_types.clone()]))
                            // {
                            //     error.entry.get_mut().insert(arg_types);
                            // }
                        }

                        // Non-parametric procedure: just type-check the arguments.
                        let fn_type_id = self.node_types[fn_decl.lhs as usize].first();
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
            _ => unreachable!("Definition not found: {}", self.tree.name(callee_id)),
        }
        Ok(())
    }

    fn check_arguments(
        &mut self,
        argument_types: &Vec<(NodeId, TypeIds)>,
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
            for arg_type in arg_type_ids.all() {
                let param_type = parameter_types[parameter_index];
                parameter_index += 1;
                if arg_type != param_type {
                    // Check if untyped argument is compatible with parameter.
                    if is_integer(param_type) && arg_type == BuiltInType::IntegerLiteral as TypeId {
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
            self.node_types[*node_id as usize] = Single(type_id);
        }
        CallResult::Ok
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
    fn set_node_type(&mut self, index: u32, type_ids: TypeIds) {
        self.node_types[index as usize] = type_ids;
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
    type_id == BuiltInType::Integer as TypeId || type_id == BuiltInType::Unsigned8 as TypeId
}

fn infer_type(annotation_type: TypeId, value_type: TypeId) -> Result<TypeIds> {
    let void = BuiltInType::Void as TypeId;
    if annotation_type != void
        && (value_type == void || value_type == BuiltInType::IntegerLiteral as TypeId)
    {
        // Explicit type based on annotation.
        Ok(Single(annotation_type))
    } else if annotation_type == void && value_type == BuiltInType::IntegerLiteral as TypeId {
        Ok(Single(BuiltInType::Integer as TypeId))
    } else if annotation_type == void && value_type != void {
        // Infer type based on rvalue.
        Ok(Single(value_type))
    } else if annotation_type != value_type {
        Err(Diagnostic::error().with_message("annotation type doesn't match"))
    } else {
        Ok(Single(annotation_type))
    }
}
