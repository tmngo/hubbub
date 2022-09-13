use crate::{
    analyze::{Definition, Lookup},
    parse::{Node, NodeId, Tag, Tree},
};
use color_eyre::{
    eyre::{eyre, Result},
    Section,
};
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

#[derive(Debug)]
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
    },
    Function {
        parameters: Vec<TypeId>,
        returns: Vec<TypeId>,
    },
    Pointer {
        typ: TypeId,
    },
    Struct {
        fields: Vec<TypeId>,
    },
    TypeParameter,
}

impl Type {}

pub type TypeId = usize;

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

pub struct Typechecker<'a> {
    tree: &'a Tree,
    definitions: &'a mut HashMap<NodeId, Definition>,
    pub types: Vec<Type>,
    type_definitions: HashMap<TypeId, NodeId>,
    pub error_reports: Vec<TypeError>,
    pub node_types: Vec<TypeId>,
    array_types: HashMap<(TypeId, usize), TypeId>,
    pointer_types: HashMap<TypeId, TypeId>,
    type_parameters: HashMap<NodeId, HashSet<Vec<TypeId>>>,
}

impl<'a> Typechecker<'a> {
    pub fn new(tree: &'a Tree, definitions: &'a mut HashMap<u32, Definition>) -> Self {
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
            tree,
            definitions,
            type_definitions: HashMap::new(),
            types,
            error_reports: vec![],
            node_types: vec![0; tree.nodes.len()],
            array_types: HashMap::new(),
            pointer_types: HashMap::new(),
            type_parameters: HashMap::new(),
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
        if self.error_reports.len() > 0 {
            let err = eyre!(
                "There were {} typechecking errors.",
                self.error_reports.len()
            );
            let result = Err(err);
            let mut type_errors = Vec::new();
            type_errors.append(&mut self.error_reports);
            return type_errors.into_iter().fold(result, |report, e| {
                report.error(e).suggestion("Try doing ...")
            });
        }
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
                let type_id = self.infer_node(node.lhs)?;
                self.type_definitions.insert(type_id, index);
            }
            Tag::Struct => {
                let type_id = self.infer_node(index)?;
                self.type_definitions.insert(type_id, index);
            }
            _ => {}
        };
        Ok(())
    }

    ///
    fn infer_range(&mut self, node: &Node) -> Result<TypeId> {
        for i in node.lhs..node.rhs {
            self.infer_node(self.tree.indices[i as usize])?;
            // if let Err(report) = result {
            //     self.error_reports
            //         .push(TypeError(report.root_cause().to_string()));
            // }
        }
        Ok(0)
    }

    ///
    fn infer_node(&mut self, node_id: NodeId) -> Result<TypeId> {
        if node_id == 0 {
            return Ok(0);
        }
        if self.node_types[node_id as usize] != 0 {
            return Ok(self.node_types[node_id as usize]);
        }
        let node = self.tree.node(node_id);
        println!("[{}] - {:?}", node_id, node.tag);
        let result = match node.tag {
            Tag::Access => {
                let ltype = self.infer_node(node.lhs)?;
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
                if let (Type::Struct { fields }, Definition::User(field_index)) = (
                    &self.types[ltype as usize],
                    self.definitions
                        .get(&node.rhs)
                        .expect("field index not defined"),
                ) {
                    fields[*field_index as usize]
                } else {
                    return Err(eyre!("cannot access non-struct"));
                }
            }
            Tag::Address => {
                let value_type = self.infer_node(node.lhs)?;
                self.add_pointer_type(value_type)
            }
            Tag::Add
            | Tag::Div
            | Tag::Mul
            | Tag::Sub
            | Tag::BitwiseAnd
            | Tag::BitwiseOr
            | Tag::BitwiseShiftL
            | Tag::BitwiseShiftR
            | Tag::BitwiseXor => self.infer_binary_node(node.lhs, node.rhs)?,
            Tag::Assign => {
                let ltype = self.infer_node(node.lhs)?;
                let rtype = self.infer_node(node.rhs)?;
                if node.rhs != 0 && ltype != rtype {
                    return Err(eyre!(
                        "mismatched types in assignment: expected {:?}, got {:?}",
                        self.types[ltype],
                        self.types[rtype]
                    ));
                }
                0
            }
            Tag::Block
            | Tag::Expressions
            | Tag::IfElse
            | Tag::Module
            | Tag::Parameters
            | Tag::Return => self.infer_range(&node)?,
            Tag::Call => {
                let ltype = self.infer_node(node.lhs)?;
                self.infer_node(node.rhs)?;
                if let Type::Function {
                    parameters: _,
                    returns,
                } = &self.types[ltype]
                {
                    if returns.len() > 0 {
                        returns[0]
                    } else {
                        0
                    }
                } else {
                    0
                }
            }
            Tag::Dereference => {
                let pointer_type = self.infer_node(node.lhs)?;
                if let Type::Pointer { typ } = self.types[pointer_type] {
                    typ
                } else {
                    return Err(eyre!(
                        "[line {}] type \"{:?}\" cannot be dereferenced",
                        self.tree.node_token_line(node_id),
                        self.types[pointer_type]
                    ));
                }
            }
            Tag::Field => self.infer_node(self.tree.node_index(node.rhs))?,
            Tag::FunctionDecl => {
                // Prototype
                let fn_type = self.infer_node(node.lhs)?;
                // Immediately set the declaration's type to handle recursion.
                self.set_node_type(node_id, fn_type);
                // Body
                self.infer_node(node.rhs)?;
                fn_type
            }
            Tag::Equality | Tag::Greater | Tag::Inequality | Tag::Less => {
                self.infer_binary_node(node.lhs, node.rhs)?;
                TypeIndex::Boolean as TypeId
            }
            Tag::Identifier => {
                let decl = self.definitions.get(&node_id);
                if let Some(lookup) = decl {
                    match lookup {
                        Definition::User(decl_index) => self.infer_node(*decl_index)?,
                        Definition::BuiltIn(type_index) => *type_index as TypeId,
                        _ => 0,
                    }
                } else {
                    0
                }
            }
            Tag::IntegerLiteral => TypeIndex::Integer as TypeId,
            Tag::True | Tag::False => TypeIndex::Boolean as TypeId,
            Tag::ParametricPrototype => {
                // Prototype
                self.infer_node(node.rhs)?
            }
            Tag::Prototype => {
                let mut parameters = Vec::new();
                let mut returns = Vec::new();
                self.infer_node(node.lhs)?; // parameters
                let return_type = self.infer_node(node.rhs)?; // returns
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
                } else if rets.tag == Tag::Identifier {
                    returns.push(return_type);
                }
                self.add_type(Type::Function {
                    parameters,
                    returns,
                })
            }
            Tag::Struct => {
                let mut fields = Vec::new();
                for i in node.lhs..node.rhs {
                    self.infer_node(self.tree.node_index(i))?;
                }
                for i in node.lhs..node.rhs {
                    let ni = self.tree.node_index(i) as usize;
                    fields.push(self.node_types[ni]);
                }
                self.add_type(Type::Struct { fields })
            }
            Tag::Subscript => {
                let array_type = self.infer_node(node.lhs)?;
                self.infer_node(node.rhs)?;
                if let Type::Array { typ, .. } = self.types[array_type] {
                    typ
                } else {
                    return Err(eyre!(
                        "[line {}] type \"{:?}\" cannot be indexed",
                        self.tree.node_token_line(node_id),
                        self.types[array_type]
                    ));
                }
            }
            Tag::Type => {
                let definition = self.definitions.get(&node_id);
                let base_type = if let Some(lookup) = definition {
                    match lookup {
                        Definition::User(definition_id) => self.infer_node(*definition_id)?,
                        Definition::BuiltIn(type_index) => *type_index as TypeId,
                        _ => 0,
                    }
                } else {
                    return Err(eyre!("Undefined type"));
                };
                println!("base type: {:?}", base_type);
                match base_type {
                    6 => {
                        println!("{}", crate::format_red!("Checking Array type"));
                        // Expect two type parameters.
                        assert_eq!(node.tag, Tag::Type);
                        if node.rhs - node.lhs != 2 {
                            return Err(eyre!(
                                "Expected 2 type parameters, got {}.",
                                node.rhs - node.lhs
                            ));
                        }
                        let ni = self.tree.node_index(node.lhs);
                        let value_type = self.infer_node(ni)?;
                        let ni = self.tree.node_index(node.lhs + 1);
                        let length_node = self.tree.node(ni);
                        assert_eq!(
                            length_node.tag,
                            Tag::IntegerLiteral,
                            "The length of an Array must be an integer literal."
                        );
                        let token_str = self.tree.node_lexeme(ni);
                        dbg!(token_str);
                        let length = token_str.parse::<i64>().unwrap();
                        let array_type = self.add_array_type(value_type, length as usize);
                        array_type
                    }
                    8 => {
                        println!("{}", crate::format_red!("Checking Pointer type"));
                        // Expect one type parameter
                        if node.rhs - node.lhs != 1 {
                            return Err(eyre!(
                                "Expected 1 type parameter, got {}.",
                                node.rhs - node.lhs
                            ));
                        }
                        let ni = self.tree.node_index(node.lhs);
                        let value_type = self.infer_node(ni)?;
                        println!("value type: {:?}", self.types[value_type]);

                        // Map from type parameter T to Pointer{T}
                        let ptr_type = self.add_pointer_type(value_type);
                        dbg!(ptr_type);
                        ptr_type
                    }
                    _ => return Err(eyre!("Undefined type")),
                }
            }
            Tag::VariableDecl => {
                // lhs: type-expr
                // rhs: init-expr
                let ltype = self.infer_node(node.lhs)?;
                let rtype = self.infer_node(node.rhs)?;
                if ltype != 0 && rtype == 0 {
                    ltype
                } else if ltype == 0 && rtype != 0 {
                    rtype
                } else if ltype != rtype {
                    return Err(eyre!("annotation type doesn't match"));
                } else {
                    ltype
                }
            }
            _ => {
                self.infer_node(node.lhs)?;
                self.infer_node(node.rhs)?;
                0
            }
        };
        self.set_node_type(node_id, result);
        return Ok(self.node_types[node_id as usize]);
    }

    ///
    fn infer_binary_node(&mut self, lhs: u32, rhs: u32) -> Result<TypeId> {
        let ltype = self.infer_node(lhs)?;
        let rtype = self.infer_node(rhs)?;
        if ltype != rtype {
            return Err(eyre!(
                "mismatched types: lhs is {:?}, rhs is {:?}",
                self.types[ltype],
                self.types[rtype]
            ));
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
            }),
            Err(err) => *err.entry.get(),
        }
    }

    fn add_pointer_type(&mut self, value_type: TypeId) -> TypeId {
        match self.pointer_types.try_insert(value_type, self.types.len()) {
            Ok(_) => self.add_type(Type::Pointer { typ: value_type }),
            Err(err) => *err.entry.get(),
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
