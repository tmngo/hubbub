use crate::{
    analyze::{BuiltInType, Definition},
    parse::{Node, NodeId, Tag, Tree},
    workspace::{Result, Workspace},
};
use codespan_reporting::diagnostic::Diagnostic;
use smallvec::{smallvec, SmallVec};
use std::collections::{HashMap, HashSet};
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

#[derive(Debug, Clone)]
pub enum Type {
    Void,
    Boolean,
    Integer,
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

impl Type {}

pub type TypeId = usize;

// pub enum TypeReference {
//     Mono(TypeId),
//     Poly(Vec<TypeId>),
// }

pub enum TypeIndex {
    Void,     // 0
    Boolean,  // 1
    Integer,  // 2
    Float,    // 3
    String,   // 4
    Type,     // 5
    Array,    // 6
    Function, // 7
    Pointer,  // 8
    Struct,   // 9
}

type TypeMap<P = Vec<TypeId>> = HashMap<P, TypeId>;

pub struct Typechecker<'a> {
    workspace: &'a mut Workspace,
    tree: &'a Tree,
    definitions: &'a mut HashMap<NodeId, Definition>,
    overload_sets: &'a HashMap<NodeId, Vec<NodeId>>,
    pub types: Vec<Type>,
    type_definitions: HashMap<TypeId, NodeId>,
    pub error_reports: Vec<TypeError>,
    pub node_types: Vec<TypeId>,
    array_types: TypeMap<(TypeId, usize)>,
    pointer_types: TypeMap<TypeId>,
    polymorphic_types: HashMap<NodeId, TypeMap>,
    type_parameters: HashMap<NodeId, HashSet<Vec<TypeId>>>,
    current_parameters: Vec<TypeId>,
    // This will have to be updated for nested struct definitions.
    current_struct_id: NodeId,
}

impl<'a> Typechecker<'a> {
    pub fn new(
        workspace: &'a mut Workspace,
        tree: &'a Tree,
        definitions: &'a mut HashMap<u32, Definition>,
        overload_sets: &'a HashMap<NodeId, Vec<NodeId>>,
    ) -> Self {
        let types = vec![
            Type::Void,
            Type::Boolean,
            Type::Integer,
            Type::Float,
            Type::String,
            Type::Type,
            // Type::Pointer {
            //     typ: TypeIndex::Integer as TypeId,
            // },
            // alloc
            // Type::Function {
            //     parameters: vec![TypeIndex::Integer as usize],
            //     returns: vec![6],
            // },
        ];
        Self {
            workspace,
            tree,
            definitions,
            overload_sets,
            type_definitions: HashMap::new(),
            types,
            error_reports: vec![],
            node_types: vec![0; tree.nodes.len()],
            array_types: HashMap::new(),
            pointer_types: HashMap::new(),
            polymorphic_types: HashMap::new(),
            type_parameters: HashMap::new(),
            current_parameters: vec![],
            current_struct_id: 0,
        }
    }

    pub fn results(self) -> (Vec<Type>, Vec<usize>, HashMap<NodeId, HashSet<Vec<TypeId>>>) {
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

    fn infer_declaration_type(&mut self, index: u32) -> Result<()> {
        if index == 0 || self.node_types[index as usize] != 0 {
            return Ok(());
        }
        let node = self.tree.node(index);
        match node.tag {
            Tag::FunctionDecl => {
                // prototype
                let type_id = self.infer_node(node.lhs)?[0];
                self.type_definitions.insert(type_id, index);
            }
            Tag::Struct => {
                let type_id = self.infer_node(index)?[0];
                self.type_definitions.insert(type_id, index);
            }
            _ => {}
        };
        Ok(())
    }

    ///
    fn infer_range(&mut self, node: &Node) -> Result<TypeId> {
        for i in node.lhs..node.rhs {
            let result = self.infer_node(self.tree.indices[i as usize]);
            if let Err(diagnostic) = result {
                self.workspace.diagnostics.push(diagnostic);
            }
        }
        Ok(0)
    }

    ///
    fn infer_node(&mut self, node_id: NodeId) -> Result<SmallVec<[TypeId; 8]>> {
        if node_id == 0 {
            return Ok(smallvec![0]);
        }
        if self.node_types[node_id as usize] != 0 {
            return Ok(smallvec![self.node_types[node_id as usize]]);
        }
        let node = self.tree.node(node_id);
        let mut inferred_node_ids = vec![];
        let mut inferred_type_ids = vec![];
        // println!("[{}] - {:?}", node_id, node.tag);
        let result = match node.tag {
            Tag::Access => {
                let ltype = self.infer_node(node.lhs)?[0];
                // Module access.
                if ltype == 0 {
                    return self.infer_node(node.rhs);
                }

                if let Some(&type_definition) = self.type_definitions.get(&ltype) {
                    let struct_decl = self.tree.node(type_definition);
                    assert_eq!(struct_decl.tag, Tag::Struct);
                    for i in struct_decl.lhs..struct_decl.rhs {
                        let field_id = self.tree.node_index(i);
                        if self.tree.name(field_id) == self.tree.name(node.rhs) {
                            self.definitions.insert(node_id, Definition::User(field_id));
                            let field = self.tree.node(field_id);
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
                    fields[*field_index as usize]
                } else {
                    return Err(Diagnostic::error().with_message("cannot access non-struct"));
                }
            }
            Tag::Address => {
                let value_type = self.infer_node(node.lhs)?[0];
                self.add_pointer_type(value_type)
            }
            Tag::Add => {
                let ltype = self.infer_node(node.lhs)?[0];
                self.infer_node(node.rhs)?;
                ltype
            }
            Tag::Div
            | Tag::Mul
            | Tag::Sub
            | Tag::BitwiseAnd
            | Tag::BitwiseOr
            | Tag::BitwiseShiftL
            | Tag::BitwiseShiftR
            | Tag::BitwiseXor => self.infer_binary_node(node)?,
            Tag::Assign => {
                let ltype = self.infer_node(node.lhs)?[0];
                let rtype = self.infer_node(node.rhs)?[0];
                if node.rhs != 0 && ltype != rtype {
                    return Err(Diagnostic::error().with_message(format!(
                        "mismatched types in assignment: expected {:?}, got {:?}",
                        self.types[ltype], self.types[rtype]
                    )));
                }
                0
            }
            Tag::Block
            | Tag::Expressions
            | Tag::IfElse
            | Tag::Module
            | Tag::Parameters
            | Tag::Return => self.infer_range(node)?,
            Tag::Call => {
                // Argument expressions
                self.infer_node(node.rhs)?;
                let expressions = self.tree.rchild(node);

                // Callee
                let callee_id = node.lhs;
                let definition = self.definitions.get(&callee_id).unwrap_or_else(|| {
                    panic!("Definition not found: {}", self.tree.name(callee_id))
                });
                let (definition_id, is_overload) = match definition {
                    Definition::BuiltIn(id) => (*id as u32, false),
                    Definition::User(id) | Definition::Foreign(id) | Definition::Resolved(id) => {
                        (*id, false)
                    }
                    Definition::Overload(id) => (*id, true),
                    Definition::NotFound => {
                        unreachable!("Definition not found: {}", self.tree.name(callee_id))
                    }
                };

                if is_overload {
                    let overload_set = self.overload_sets.get(&definition_id).unwrap();
                    let mut match_found = false;

                    let mut argument_types = vec![];
                    for i in expressions.lhs..expressions.rhs {
                        let arg_id = self.tree.node_index(i);
                        let arg_type = self.node_types[arg_id as usize];
                        argument_types.push(arg_type);
                    }

                    'outer: for &fn_decl_id in overload_set {
                        let fn_decl = self.tree.node(fn_decl_id);
                        let prototype = self.tree.node(fn_decl.lhs);
                        let parameters = self.tree.lchild(prototype);
                        assert_eq!(parameters.tag, Tag::Parameters);

                        if (parameters.rhs - parameters.lhs) as usize != argument_types.len() {
                            continue 'outer;
                        }

                        for i in parameters.lhs..parameters.rhs {
                            let param_id = self.tree.node_index(i);
                            let param_type = self.node_types[param_id as usize];
                            let arg_type = argument_types[(i - parameters.lhs) as usize];
                            if arg_type != param_type {
                                continue 'outer;
                            }
                        }

                        match_found = true;
                        self.definitions
                            .insert(callee_id, Definition::Resolved(fn_decl_id));
                        self.definitions
                            .insert(fn_decl_id, Definition::Resolved(fn_decl_id));
                    }
                    if !match_found {
                        return Err(
                            Diagnostic::error().with_message("failed to find matching overload")
                        );
                    }
                } else {
                    let fn_decl = self.tree.node(definition_id);
                    let prototype = self.tree.node(fn_decl.lhs);
                    match prototype.tag {
                        Tag::ParametricPrototype => {
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
                        Tag::Prototype => {
                            // Non-parametric procedure: just type-check the arguments.
                            let mut argument_types = vec![];
                            for i in expressions.lhs..expressions.rhs {
                                let arg_id = self.tree.node_index(i);
                                let arg_type = self.node_types[arg_id as usize];
                                argument_types.push(arg_type);
                            }

                            let parameters = self.tree.lchild(prototype);
                            assert_eq!(parameters.tag, Tag::Parameters);

                            if (parameters.rhs - parameters.lhs) as usize != argument_types.len() {
                                return Err(Diagnostic::error().with_message(format!(
                                    "invalid function call: expected {:?} arguments, got {:?}",
                                    parameters.rhs - parameters.lhs,
                                    argument_types.len()
                                )));
                            }

                            for i in parameters.lhs..parameters.rhs {
                                let param_id = self.tree.node_index(i);
                                let param_type = self.node_types[param_id as usize];
                                let arg_type = argument_types[(i - parameters.lhs) as usize];
                                if arg_type != param_type {
                                    return Err(Diagnostic::error().with_message(format!(
                                        "mismatched types in function call: expected {:?} argument, got {:?}",
                                        self.types[param_type],
                                        self.types[arg_type])
                                    ));
                                }
                            }
                        }
                        _ => {
                            println!(
                                "{}",
                                crate::format_red!("TODO: skipped typechecking of builtin fn")
                            );
                        }
                    }
                }

                let ltype = self.infer_node(callee_id)?[0];

                if let Type::Function {
                    parameters: _,
                    returns,
                } = &self.types[ltype]
                {
                    if !returns.is_empty() {
                        returns[0]
                    } else {
                        0
                    }
                } else {
                    0
                }
            }
            Tag::Dereference => {
                let pointer_type = self.infer_node(node.lhs)?[0];
                if let Type::Pointer { typ, .. } = self.types[pointer_type] {
                    typ
                } else {
                    return Err(Diagnostic::error().with_message(format!(
                        "[line {}] type \"{:?}\" cannot be dereferenced",
                        self.tree.node_token_line(node_id),
                        self.types[pointer_type]
                    )));
                }
            }
            Tag::Field => self.infer_node(self.tree.node_index(node.rhs))?[0],
            Tag::FunctionDecl => {
                // Prototype
                let fn_type = self.infer_node(node.lhs)?[0];
                // Immediately set the declaration's type to handle recursion.
                self.set_node_type(node_id, fn_type);
                // Body
                self.infer_node(node.rhs)?;
                fn_type
            }
            Tag::Equality | Tag::Greater | Tag::Inequality | Tag::Less => {
                self.infer_binary_node(node)?;
                TypeIndex::Boolean as TypeId
            }
            Tag::Identifier => {
                // The type of an identifier is the type of its definition.
                let decl = self.definitions.get(&node_id);
                if let Some(lookup) = decl {
                    match lookup {
                        Definition::User(decl_id) | Definition::Resolved(decl_id) => {
                            self.infer_node(*decl_id)?[0]
                        }
                        Definition::BuiltIn(type_id) => *type_id as TypeId,
                        Definition::Overload(_) => unreachable!(),
                        _ => 0,
                    }
                } else {
                    0
                }
            }
            Tag::IntegerLiteral => TypeIndex::Integer as TypeId,
            Tag::True | Tag::False => TypeIndex::Boolean as TypeId,
            Tag::ParametricPrototype => {
                // Type parameters
                let type_parameters = self.tree.node(node.lhs);
                for i in type_parameters.lhs..type_parameters.rhs {
                    let node_id = self.tree.node_index(i);
                    self.add_type(Type::Parameter {
                        index: (i - type_parameters.lhs) as usize,
                    });
                    self.set_node_type(node_id, self.types.len() - 1);
                }
                // Prototype
                self.infer_node(node.rhs)?[0]
            }
            Tag::Prototype => {
                let mut parameters = Vec::new();
                let mut returns = Vec::new();
                self.infer_node(node.lhs)?; // parameters
                let return_type = self.infer_node(node.rhs)?[0]; // returns
                let params = self.tree.node(node.lhs);
                for i in params.lhs..params.rhs {
                    let ni = self.tree.node_index(i) as usize;
                    parameters.push(self.node_types[ni]);
                }
                let rets = self.tree.node(node.rhs);
                if rets.tag == Tag::Expressions {
                    for i in rets.lhs..rets.rhs {
                        let ni = self.tree.node_index(i) as usize;
                        returns.push(self.node_types[ni]);
                    }
                } else if rets.tag == Tag::Identifier || rets.tag == Tag::Type {
                    returns.push(return_type);
                }
                self.add_type(Type::Function {
                    parameters,
                    returns,
                })
            }
            Tag::Struct => {
                if node.lhs != 0 {
                    let type_parameters = self.tree.node(node.lhs);
                    for i in type_parameters.lhs..type_parameters.rhs {
                        let node_id = self.tree.node_index(i);
                        let param_type = self.add_type(Type::Parameter {
                            index: (i - type_parameters.lhs) as usize,
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
                self.add_type(Type::Struct { fields, is_generic })
            }
            Tag::Subscript => {
                let array_type = self.infer_node(node.lhs)?[0];
                self.infer_node(node.rhs)?;
                if let Type::Array { typ, .. } = self.types[array_type] {
                    typ
                } else {
                    return Err(Diagnostic::error().with_message(format!(
                        "[line {}] type \"{:?}\" cannot be indexed",
                        self.tree.node_token_line(node_id),
                        self.types[array_type]
                    )));
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
                                return Err(Diagnostic::error().with_message(format!(
                                    "Expected 2 type parameters, got {}.",
                                    node.rhs - node.lhs
                                )));
                            }
                            let ni = self.tree.node_index(node.lhs);
                            let value_type = self.infer_node(ni)?[0];
                            let ni = self.tree.node_index(node.lhs + 1);
                            let length_node = self.tree.node(ni);
                            assert_eq!(
                                length_node.tag,
                                Tag::IntegerLiteral,
                                "The length of an Array must be an integer literal."
                            );
                            let token_str = self.tree.node_lexeme(ni);
                            let length = token_str.parse::<i64>().unwrap();
                            self.add_array_type(value_type, length as usize)
                        }
                        BuiltInType::Pointer => {
                            // Map concrete type argument to a pointer type.
                            // Expect one type parameter
                            if node.rhs - node.lhs != 1 {
                                return Err(Diagnostic::error().with_message(format!(
                                    "Expected 1 type parameter, got {}.",
                                    node.rhs - node.lhs
                                )));
                            }
                            let ni = self.tree.node_index(node.lhs);
                            let value_type = self.infer_node(ni)?[0];
                            self.add_pointer_type(value_type)
                        }
                        _ => {
                            return Err(Diagnostic::error().with_message("Undefined built-in type"))
                        }
                    },
                    Definition::User(id) => {
                        // Map concrete type arguments to a struct type.
                        let struct_decl_id = *id;
                        self.current_struct_id = *id;
                        let current_parameter_count = self.current_parameters.len();
                        for i in self.tree.range(node) {
                            let ni = self.tree.node_index(i);
                            let param_type = self.infer_node(ni)?[0];
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
                let identifiers = self.tree.node(node.token);

                let annotation = self.infer_node(node.lhs)?[0];
                self.infer_node(node.rhs)?;

                let mut rtypes = SmallVec::<[usize; 8]>::new();
                let rvalues = self.tree.node(node.rhs);
                match rvalues.tag {
                    Tag::Expressions => {
                        for i in rvalues.lhs..rvalues.rhs {
                            let ni = self.tree.node_index(i);
                            let ti = self.node_types[ni as usize];
                            rtypes.push(ti);
                        }
                    }
                    _ => {
                        let ni = node.rhs;
                        let ti = self.node_types[ni as usize];
                        rtypes.push(ti);
                    }
                }

                // dbg!(&rtypes);
                // dbg!(annotation);

                match identifiers.tag {
                    Tag::Expressions => {
                        for i in identifiers.lhs..identifiers.rhs {
                            let ni = self.tree.node_index(i);
                            inferred_node_ids.push(ni);
                            let rtype = rtypes[(i - identifiers.lhs) as usize];
                            let inferred_type = infer_type(annotation, rtype)?;
                            inferred_type_ids.push(inferred_type);
                        }
                    }
                    Tag::Identifier => {
                        let ni = node.token;
                        inferred_node_ids.push(ni);
                        let rtype = rtypes[0];
                        let inferred_type = infer_type(annotation, rtype)?;
                        inferred_type_ids.push(inferred_type);
                    }
                    _ => unreachable!(),
                }
                0
                // infer_type(annotation, rtypes[0])?
            }
            _ => {
                self.infer_node(node.lhs)?;
                self.infer_node(node.rhs)?;
                0
            }
        };
        for (&ni, &ti) in inferred_node_ids.iter().zip(inferred_type_ids.iter()) {
            self.set_node_type(ni, ti);
        }
        self.set_node_type(node_id, result);
        Ok(smallvec![self.node_types[node_id as usize]])
    }

    ///
    fn infer_binary_node(&mut self, node: &Node) -> Result<TypeId> {
        let ltype = self.infer_node(node.lhs)?[0];
        let rtype = self.infer_node(node.rhs)?[0];
        if ltype != rtype {
            return Err(Diagnostic::error()
                .with_message(format!(
                    "mismatched types: left is \"{:?}\", right is \"{:?}\"",
                    self.types[ltype], self.types[rtype]
                ))
                .with_labels(vec![self.tree.label(node.token)]));
        }
        Ok(ltype)
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

    fn is_generic(&self, type_id: TypeId) -> bool {
        match self.types[type_id] {
            Type::Parameter { .. } => true,
            Type::Array { is_generic, .. } | Type::Struct { is_generic, .. } => is_generic,
            _ => false,
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
                self.add_type(Type::Struct { fields, is_generic })
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

    ///
    fn set_node_type(&mut self, index: u32, type_index: TypeId) {
        self.node_types[index as usize] = type_index;
    }

    ///
    pub fn print(&self) {
        for i in 0..self.types.len() {
            println!("Type [{}]: {:?}", { i }, self.types[i]);
        }
        for i in 1..self.tree.nodes.len() {
            println!(
                "[{i}] {:?}: {:?}",
                self.tree.node(i as u32).tag,
                self.types[self.node_types[i] as usize]
            );
        }
    }
}

fn infer_type(annotation_type: TypeId, value_type: TypeId) -> Result<TypeId> {
    if annotation_type != 0 && value_type == 0 {
        // Explicit type based on annotation.
        Ok(annotation_type)
    } else if annotation_type == 0 && value_type != 0 {
        // Infer type based on rvalue.
        Ok(value_type)
    } else if annotation_type != value_type {
        Err(Diagnostic::error().with_message("annotation type doesn't match"))
    } else {
        Ok(annotation_type)
    }
}
