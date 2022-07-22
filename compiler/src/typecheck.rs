use crate::analyze::Lookup;
use crate::parse::{Node, Tag, Tree};
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

#[derive(Debug)]
pub enum Type {
    Void,
    Boolean,
    Integer,
    Float,
    String,
    Type,
    Array {
        length: usize,
        typ: usize,
    },
    Function {
        parameters: Vec<usize>,
        returns: Vec<usize>,
    },
    Pointer {
        typ: usize,
    },
    Struct {
        fields: Vec<usize>,
    },
}

impl Type {}

type TypeId = usize;

pub enum TypeIndex {
    Void,
    Boolean,
    Integer,
    Float,
    String,
    Type,
}

pub struct Typechecker<'a> {
    tree: &'a Tree,
    definitions: &'a HashMap<u32, Lookup>,
    pub types: Vec<Type>,
    pub node_types: Vec<usize>,
}

impl<'a> Typechecker<'a> {
    pub fn new(tree: &'a Tree, definitions: &'a HashMap<u32, Lookup>) -> Self {
        let types = vec![
            Type::Void,
            Type::Boolean,
            Type::Integer,
            Type::Float,
            Type::String,
            Type::Type,
        ];
        Self {
            tree,
            definitions,
            types,
            node_types: vec![0; tree.nodes.len()],
        }
    }

    ///
    pub fn check(&mut self) {
        let root = &self.tree.node(0);
        // Resolve root declarations.
        for i in root.lhs..root.rhs {
            let module_id = self.tree.node_index(i);
            let module = self.tree.node(module_id);
            for i in module.lhs..module.rhs {
                let decl_id = self.tree.node_index(i);
                self.infer_declaration_type(decl_id);
            }
        }
        // self.print();
        self.infer_range(root);
    }

    fn infer_declaration_type(&mut self, index: u32) {
        if index == 0 || self.node_types[index as usize] != 0 {
            return;
        }
        let node = self.tree.node(index);
        match node.tag {
            Tag::FunctionDecl => {
                // prototype
                self.infer_node(node.lhs);
            }
            Tag::Struct => {
                self.infer_node(index);
            }
            _ => {}
        };
    }

    ///
    fn infer_range(&mut self, node: &Node) -> TypeId {
        for i in node.lhs..node.rhs {
            self.infer_node(self.tree.indices[i as usize]);
        }
        0
    }

    ///
    fn infer_node(&mut self, index: u32) -> TypeId {
        if index == 0 {
            return 0;
        }
        if self.node_types[index as usize] != 0 {
            return self.node_types[index as usize];
        }
        let node = self.tree.node(index);
        println!("[{}] - {:?}", index, node.tag);
        let result = match node.tag {
            Tag::Access => {
                let ltype = self.infer_node(node.lhs);
                if let (Type::Struct { fields }, Lookup::Defined(field_index)) = (
                    &self.types[ltype as usize],
                    self.definitions.get(&node.rhs).unwrap(),
                ) {
                    fields[*field_index as usize]
                } else {
                    panic!()
                }
            }
            Tag::Add
            | Tag::Div
            | Tag::Mul
            | Tag::Sub
            | Tag::BitwiseAnd
            | Tag::BitwiseOr
            | Tag::BitwiseXor => self.infer_binary_node(node.lhs, node.rhs),
            Tag::Block | Tag::Expressions | Tag::IfElse | Tag::Module | Tag::Parameters => {
                self.infer_range(&node)
            }
            Tag::Call => {
                let ltype = self.infer_node(node.lhs);
                self.infer_node(node.rhs);
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
            Tag::Field => {
                println!("Field: {:?}", self.tree.node_lexeme(node.lhs));
                self.infer_node(node.lhs)
            }
            Tag::FunctionDecl => {
                // Prototype
                let fn_type = self.infer_node(node.lhs);
                // Immediately set the declaration's type to handle recursion.
                self.set_node_type(index, fn_type);
                // Body
                self.infer_node(node.rhs);
                fn_type
            }
            Tag::Equality | Tag::Greater | Tag::Inequality | Tag::Less => {
                self.infer_binary_node(node.lhs, node.rhs);
                TypeIndex::Boolean as TypeId
            }
            Tag::Identifier => {
                let decl = self.definitions.get(&index);
                println!("{:?}", decl);
                if let Some(lookup) = decl {
                    match lookup {
                        Lookup::Defined(decl_index) => {
                            println!("decl_index: {}", decl_index);
                            self.infer_node(*decl_index)
                        }
                        Lookup::BuiltIn(type_index) => {
                            println!("type_index: {}", type_index);
                            *type_index as TypeId
                        }
                        _ => 0,
                    }
                } else {
                    0
                }
            }
            Tag::IntegerLiteral => TypeIndex::Integer as TypeId,
            Tag::Prototype => {
                let mut parameters = Vec::new();
                let mut returns = Vec::new();
                self.infer_node(node.lhs); // parameters
                let return_type = self.infer_node(node.rhs); // returns
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
                println!("Struct: {:?}", self.tree.node_lexeme(index));
                let mut fields = Vec::new();
                for i in node.lhs..node.rhs {
                    self.infer_node(self.tree.indices[i as usize]);
                }
                for i in node.lhs..node.rhs {
                    let ni = self.tree.node_index(i) as usize;
                    fields.push(self.node_types[ni]);
                }
                self.add_type(Type::Struct { fields })
            }
            Tag::VariableDecl => {
                // lhs: type-expr
                // rhs: init-expr
                let ltype = self.infer_node(node.lhs);
                let rtype = self.infer_node(node.rhs);
                dbg!(ltype, rtype);
                if ltype != 0 && rtype == 0 {
                    ltype
                } else if ltype == 0 && rtype != 0 {
                    rtype
                } else if ltype != rtype {
                    panic!("annotation type doesn't match");
                } else {
                    ltype
                }
            }
            _ => {
                self.infer_node(node.lhs);
                self.infer_node(node.rhs);
                0
            }
        };
        self.set_node_type(index, result);
        return self.node_types[index as usize];
    }

    ///
    fn infer_binary_node(&mut self, lhs: u32, rhs: u32) -> TypeId {
        let ltype = self.infer_node(lhs);
        let rtype = self.infer_node(rhs);
        if ltype != rtype {
            println!(
                "{:?} vs {:?}",
                self.tree.node_lexeme(lhs),
                self.tree.node_lexeme(rhs),
            );
            println!(
                "{:?} ({:?}) vs {:?} ({:?})",
                self.tree.node(lhs).tag,
                self.types[ltype as usize],
                self.tree.node(rhs).tag,
                self.types[rtype as usize]
            );
            panic!("operand types don't match");
        }
        ltype
    }

    fn add_type(&mut self, typ: Type) -> TypeId {
        self.types.push(typ);
        self.types.len() - 1
    }

    ///
    fn set_node_type(&mut self, index: u32, type_index: TypeId) {
        self.node_types[index as usize] = type_index;
    }

    ///
    pub fn print(&self) {
        for t in &self.types {
            println!("{:?}", t);
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
