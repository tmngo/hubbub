use crate::parse::{Node, NodeId, Tag, Tree};
use crate::typecheck::TypeIndex;
use crate::utils::assert_size;
use colored::Colorize;
use std::collections::HashMap;
// use std::collections::HashMap::OccupiedError;
use std::fmt;

/**
 * This associates identifiers
 * Names are stored in a stack of maps.
 */
pub struct Analyzer<'a> {
    tree: &'a Tree,
    pub definitions: HashMap<NodeId, Definition>,
    struct_scopes: HashMap<u32, Scope<'a>>,
    scopes: Vec<Scope<'a>>,
    builtins: Scope<'a>,
    foreign: Scope<'a>,
    current: usize,
}

#[derive(Debug)]
pub enum Definition {
    BuiltIn(u32),
    User(NodeId),
    Foreign(u32),
    // Overload(u32),
    NotFound,
}

// Assert that Tag size <= 1 byte
pub const _ASSERT_LOOKUP_SIZE: () = assert_size::<Definition>(8);

// #[derive(PartialEq)]
// enum State {
//     Unresolved,
//     Resolving,
//     Resolved,
// }

// enum Kind {
//     Function,
//     Global,
//     Local,
//     Parameter,
//     Struct,
// }

// struct Symbol {
//     node: u32,
//     state: State,
//     kind: Kind,
// }

#[derive(Debug)]
pub struct Scope<'a> {
    symbols: HashMap<&'a str, u32>,
    parent: usize,
}

impl<'a> Scope<'a> {
    pub fn new(parent: usize) -> Self {
        Scope {
            symbols: HashMap::new(),
            parent,
        }
    }

    pub fn from<const N: usize>(arr: [(&'a str, u32); N], parent: usize) -> Self {
        Self {
            symbols: HashMap::from(arr),
            parent,
        }
    }

    pub fn get(&self, name: &str) -> Option<&u32> {
        self.symbols.get(name)
    }

    pub fn parent(&self) -> usize {
        self.parent
    }

    // pub fn try_insert(
    //     &mut self,
    //     name: &'a str,
    //     id: u32,
    // ) -> Result<&mut u32, OccupiedError<'a, &str, u32>> {
    //     self.symbols.try_insert(name, id)
    // }

    // pub fn define(sym: Symbol) {}
    // pub fn find() -> Symbol {
    //     Symbol {
    //         node: 0,
    //         state: State::Unresolved,
    //         kind: Kind::Global,
    //     }
    // }
}

pub trait Resolve {
    fn define(&self, token: u32, lhs: u32, rhs: u32, analyzer: &mut Analyzer);
}

impl<'a> Analyzer<'a> {
    pub fn new(tree: &'a Tree) -> Self {
        let builtins = Scope::from(
            [
                ("Bool", TypeIndex::Boolean as u32),
                ("I64", TypeIndex::Integer as u32),
                ("Int", TypeIndex::Integer as u32),
                ("Int32", TypeIndex::Integer as u32),
                ("Int64", TypeIndex::Integer as u32),
                ("Float", TypeIndex::Float as u32),
                ("F64", TypeIndex::Float as u32),
                ("String", TypeIndex::String as u32),
            ],
            0,
        );
        let foreign = Scope::from([("putchar", 1), ("print_int", 2)], 0);
        Self {
            tree,
            definitions: HashMap::new(),
            builtins,
            foreign,
            scopes: Vec::new(),
            struct_scopes: HashMap::new(),
            current: 0,
        }
    }

    pub fn resolve(&mut self) {
        self.enter_scope();
        let root = &self.tree.node(0);
        let first_module = self.tree.node(self.tree.node_index(root.lhs));
        println!("Collecting module decls.");
        self.collect_module_decls(first_module);
        println!("Resolving first_module");
        self.resolve_range(root);
    }

    fn collect_module_decls(&mut self, root: &Node) {
        for i in root.lhs..root.rhs {
            let index = self.tree.node_index(i);
            let node = self.tree.node(index);
            match node.tag {
                Tag::FunctionDecl => {
                    let name = self.tree.node_lexeme_offset(&node, -1);
                    println!(
                        "Declaration: {} - Node: {:?} - Token: {:?}",
                        index, node.tag, name
                    );
                    self.define_symbol(name, index);
                }
                Tag::Import => {
                    let name = self.tree.node_lexeme_offset(&node, -2);
                    println!(
                        "Import: {} - Node: {:?} - Token: {:?}",
                        index, node.tag, name
                    );
                    self.define_symbol(name, index);
                }
                Tag::Struct => {
                    let name = self.tree.node_lexeme_offset(&node, -2);
                    println!(
                        "Declaration: {} - Node: {:?} - Token: {:?}",
                        index, node.tag, name
                    );
                    self.define_symbol(name, index);
                    let mut scope = Scope::new(0);
                    for i in node.lhs..node.rhs {
                        let ni = self.tree.node_index(i);
                        let field_name = self.tree.node_lexeme(ni);
                        scope
                            .symbols
                            .try_insert(field_name, i - node.lhs)
                            .expect(&format!("Field \"{}\" is already defined.", field_name));
                        println!(
                            "Defined field \"{}\" [{}+{}].",
                            field_name,
                            node.lhs,
                            i - node.lhs
                        );
                    }
                    self.struct_scopes.insert(index, scope);
                    println!("{:?}", self.struct_scopes);
                }
                _ => {}
            };
        }
    }

    // fn resolve_symbol(&self, mut sym: Symbol) {
    //     if sym.state == State::Resolved {
    //         return;
    //     } else if sym.state == State::Resolving {
    //         return;
    //     }

    //     sym.state = State::Resolving;
    //     match self.nodes[sym.node as usize].tag {
    //         Tag::VariableDecl => {}
    //         _ => {}
    //     }
    //     sym.state = State::Resolved;
    // }

    fn resolve_range(&mut self, node: &Node) {
        for i in node.lhs..node.rhs {
            let ni = self.tree.node_index(i);
            self.resolve_node(ni);
        }
    }

    fn resolve_node(&mut self, id: NodeId) {
        if id == 0 {
            return;
        }
        let node = &self.tree.node(id);
        println!(
            "Node [{}]: {:?} / {:?} ({}, {})",
            id,
            node.tag,
            self.tree.node_token(node).tag,
            node.lhs,
            node.rhs
        );
        match node.tag {
            Tag::Access => {
                // lhs: container
                // rhs: member identifier
                // Assumes lhs is a declaration.
                self.resolve_node(node.lhs);
                // This should always return a struct definition.
                let struct_def = self.get_container_definition(node.lhs);
                // Resolve type
                dbg!(struct_def);
                if struct_def != 0 {
                    let identifier = self.tree.node(node.rhs);
                    println!("{:?}", identifier.tag);
                    let field_name = self.tree.node_lexeme(node.rhs);
                    println!("{}", field_name);
                    let struct_def_node = self.tree.node(struct_def);
                    let field_index = self
                        .struct_scopes
                        .get(&struct_def)
                        .unwrap()
                        .get(&field_name)
                        .unwrap();
                    // Map access node to field_id.
                    self.definitions.insert(
                        id,
                        Definition::User(self.tree.node_index(struct_def_node.lhs + *field_index)),
                    );
                    // Map member identifier to field_index.
                    self.definitions
                        .insert(node.rhs, Definition::User(*field_index));
                } else {
                    panic!("failed to look up lhs");
                }
            }
            Tag::Block | Tag::Module => {
                self.enter_scope();
                self.resolve_range(node);
                self.exit_scope();
            }
            Tag::Expressions | Tag::IfElse | Tag::Parameters | Tag::Struct => {
                self.resolve_range(node);
            }
            Tag::FunctionDecl => {
                self.enter_scope();
                self.resolve_node(node.lhs); // Prototype
                let body = &self.tree.node(node.rhs);
                self.resolve_range(body);
                self.exit_scope();
            }
            Tag::Field => {
                let name = self.tree.node_lexeme(id);
                self.define_symbol(name, id);
                self.resolve_node(node.lhs);
            }
            Tag::Identifier => {
                let name = self.tree.node_lexeme(id);
                let definition = self.lookup(name);
                println!(" - Looking up `{}` -> {:?}", name, definition);
                match definition {
                    Definition::User(_) => {
                        self.definitions.insert(id, definition);
                    }
                    Definition::BuiltIn(_) => {
                        self.definitions.insert(id, definition);
                    }
                    Definition::Foreign(_) => {}
                    _ => {
                        println!("cannot find value `{}` in this scope", name);
                        panic!()
                    }
                }
            }
            Tag::Import => {}
            Tag::VariableDecl => {
                let name = self.tree.node_lexeme_offset(node, -1);
                self.define_symbol(name, id);
                self.resolve_node(node.lhs);
                self.resolve_node(node.rhs);
            }
            _ => {
                self.resolve_node(node.lhs);
                self.resolve_node(node.rhs);
            }
        }
    }

    fn get_container_definition(&self, id: NodeId) -> NodeId {
        match self.tree.node(id).tag {
            Tag::Access | Tag::Identifier => {
                // identifier -> variable decl / access -> field
                let lookup = self.definitions.get(&id).expect(&format!(
                    "failed to find definition for identifier \"{:?}\".",
                    self.tree.node_lexeme(id)
                ));
                if let Definition::User(decl_id) = lookup {
                    let def_node = self.tree.node(*decl_id);
                    // variable decl -> struct decl / field -> struct decl
                    let type_lookup = self.definitions.get(&def_node.lhs).expect(&format!(
                        "failed to find struct definition for variable \"{}\".",
                        self.tree.node_lexeme(def_node.lhs)
                    ));
                    if let Definition::User(def_index) = type_lookup {
                        return *def_index;
                    }
                }
                unreachable!("invalid container");
            }
            _ => {
                unreachable!("invalid container")
            }
        }
    }

    fn define_symbol(&mut self, name: &'a str, id: u32) {
        let top = self.scopes.len() - 1;
        self.scopes[top]
            .symbols
            .try_insert(name, id)
            .expect(&format!(" - Name \"{}\" is already defined.", name));
        println!(" - Defined name \"{}\".", name.red())
    }

    fn lookup(&self, name: &str) -> Definition {
        if self.scopes.len() == 0 {
            return Definition::NotFound;
        }
        let mut scope_index = self.scopes.len() - 1;
        loop {
            // Check if the name is defined in the current scope.
            if let Some(&index) = self.scopes[scope_index].get(name) {
                return Definition::User(index);
            }
            // If the name isn't in the global scope, it's undefined.
            if scope_index == 0 {
                if let Some(&index) = self.builtins.get(name) {
                    return Definition::BuiltIn(index);
                }
                if let Some(&index) = self.foreign.get(name) {
                    return Definition::Foreign(index);
                }
                return Definition::NotFound;
            }
            // Look in the parent scope.
            scope_index = self.scopes[scope_index].parent();
        }
    }

    // fn lookup_current(&self, name: &str) -> u32 {
    //     if self.scopes.len() == 0 {
    //         return 0;
    //     }
    //     let top = self.scopes.len() - 1;
    //     self.scopes[top].get(name)
    // }

    fn enter_scope(&mut self) {
        let parent = self.current;
        self.scopes.push(Scope::new(parent));
        self.current = self.scopes.len() - 1;
        println!("enter_scope {}", self.current);
    }

    fn exit_scope(&mut self) {
        self.current = self.scopes[self.current].parent();
        println!("exit_scope {}", self.current);
    }

    // fn scope_depth(&self) -> usize {
    //     let mut depth = 0;
    //     let mut index = self.current;
    //     while index != 0 {
    //         index = self.scopes[index].parent();
    //         depth += 1;
    //     }
    //     depth
    // }
}

impl<'a> fmt::Display for Analyzer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for scope in &self.scopes {
            writeln!(f, "{:?}", scope);
        }
        writeln!(f, "{:?}", self.definitions)
    }
}
