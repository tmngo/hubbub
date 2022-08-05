use crate::parse::{Node, NodeId, Tag, Tree};
use crate::typecheck::TypeIndex;
use crate::utils::assert_size;
use color_eyre::eyre::{eyre, Result, WrapErr};
use std::collections::HashMap;
// use std::collections::HashMap::OccupiedError;
use crate::output::Color;
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

#[derive(Debug, Eq, PartialEq, Hash)]
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
    named_imports: HashMap<&'a str, usize>,
    unnamed_imports: Vec<usize>,
}

impl<'a> Scope<'a> {
    pub fn new(parent: usize) -> Self {
        Scope {
            symbols: HashMap::new(),
            parent,
            named_imports: HashMap::new(),
            unnamed_imports: Vec::new(),
        }
    }

    pub fn from<const N: usize>(arr: [(&'a str, u32); N], parent: usize) -> Self {
        Self {
            symbols: HashMap::from(arr),
            parent,
            named_imports: HashMap::new(),
            unnamed_imports: Vec::new(),
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
                ("Pointer", TypeIndex::Pointer as u32),
                ("alloc", 6),
            ],
            0,
        );
        let foreign = Scope::from(
            [
                ("putchar", 1),
                ("print_int", 2),
                ("alloc", 3),
                ("dealloc", 4),
            ],
            0,
        );
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

    pub fn resolve(&mut self) -> Result<()> {
        self.enter_scope();
        let root = &self.tree.node(0);
        let mut module_scopes = Vec::new();
        for i in root.lhs..root.rhs {
            let ni = self.tree.node_index(i);
            let module = self.tree.node(ni);
            self.enter_scope();
            module_scopes.push(self.current);
            self.collect_module_decls(module)
                .wrap_err("Failed to collect module declarations")?;
            self.exit_scope();
        }
        for (m, i) in (root.lhs..root.rhs).enumerate() {
            let ni = self.tree.node_index(i);
            let module = self.tree.node(ni);
            self.current = module_scopes[m];
            self.resolve_range(module)
                .wrap_err("Failed to resolve module definitions")?;
        }
        Ok(())
    }

    fn collect_module_decls(&mut self, root: &Node) -> Result<()> {
        for i in root.lhs..root.rhs {
            let index = self.tree.node_index(i);
            let node = self.tree.node(index);
            match node.tag {
                Tag::FunctionDecl => {
                    let name = self.tree.node_lexeme_offset(&node, -1);
                    self.define_symbol(name, index)?;
                }
                Tag::Import => {
                    let module_name = self.tree.node_lexeme_offset(&node, 1).trim_matches('"');
                    if let Some(module_index) = self.tree.get_module_index(module_name) {
                        self.scopes[self.current]
                            .unnamed_imports
                            .push(module_index + 2);
                    }
                }
                Tag::Struct => {
                    let name = self.tree.node_lexeme_offset(&node, -2);
                    self.define_symbol(name, index)?;
                    let mut scope = Scope::new(0);
                    // Collect field names
                    for i in node.lhs..node.rhs {
                        let ni = self.tree.node_index(i);
                        let identifier_id = self.tree.node(ni).lhs;
                        let field_name = self.tree.node_lexeme(identifier_id);
                        if let Err(_) = scope.symbols.try_insert(field_name, i - node.lhs) {
                            return Err(eyre!(" - Field \"{}\" is already defined.", name));
                        }
                    }
                    self.struct_scopes.insert(index, scope);
                }
                _ => {}
            };
        }
        Ok(())
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

    fn resolve_range(&mut self, node: &Node) -> Result<()> {
        for i in node.lhs..node.rhs {
            let ni = self.tree.node_index(i);
            self.resolve_node(ni)?;
        }
        Ok(())
    }

    fn resolve_node(&mut self, id: NodeId) -> Result<()> {
        if id == 0 {
            return Ok(());
        }
        let node = &self.tree.node(id);
        match node.tag {
            Tag::Access => {
                // lhs: container
                // rhs: member identifier
                // Assumes lhs is a declaration.
                self.resolve_node(node.lhs)?;
                // This should always return a struct definition.
                let struct_def = self.get_container_definition(node.lhs);
                // Resolve type
                if struct_def != 0 {
                    let field_name = self.tree.node_lexeme(node.rhs);
                    let struct_def_node = self.tree.node(struct_def);

                    let field_index = match self
                        .struct_scopes
                        .get(&struct_def)
                        .unwrap()
                        .get(&field_name)
                    {
                        Some(value) => value,
                        _ => {
                            return Err(eyre!(
                                "struct \"{}\" does not have member \"{}\".",
                                self.tree.node_lexeme_offset(struct_def_node, -2),
                                field_name
                            ));
                        }
                    };

                    // Map access node to field_id.
                    self.definitions.insert(
                        id,
                        Definition::User(self.tree.node_index(struct_def_node.lhs + *field_index)),
                    );
                    // Map member identifier to field_index.
                    self.definitions
                        .insert(node.rhs, Definition::User(*field_index));
                } else {
                    return Err(eyre!("failed to look up lhs"));
                }
            }
            Tag::Block | Tag::Module => {
                self.enter_scope();
                self.resolve_range(node)?;
                self.exit_scope();
            }
            Tag::Expressions | Tag::IfElse | Tag::Return | Tag::Struct => {
                self.resolve_range(node)?;
            }
            Tag::FunctionDecl => {
                self.enter_scope();
                self.resolve_node(node.lhs)?; // Prototype
                let body = &self.tree.node(node.rhs);
                self.resolve_range(body)?;
                self.exit_scope();
            }
            Tag::TypeParameters => {
                for i in node.lhs..node.rhs {
                    let type_parameter_id = self.tree.node_index(i);
                    let name = self.tree.node_lexeme(type_parameter_id);
                    self.define_symbol(name, type_parameter_id)?;
                }
            }
            Tag::Parameters => {
                for i in node.lhs..node.rhs {
                    let field_id = self.tree.node_index(i);
                    let field = self.tree.node(field_id);
                    let name = self.tree.node_lexeme(field.lhs);
                    self.define_symbol(name, field_id)?;
                    // Resolve type_expr
                    self.resolve_node(field_id)?;
                }
            }
            Tag::Field => {
                // Resolve type_expr
                self.resolve_node(node.rhs)?;
            }
            Tag::Identifier => {
                let name = self.tree.node_lexeme(id);
                let definition = self.lookup(name);
                match definition {
                    Definition::User(_) => {
                        self.definitions.insert(id, definition);
                    }
                    Definition::BuiltIn(_) => {
                        self.definitions.insert(id, definition);
                    }
                    Definition::Foreign(_) => {}
                    _ => {
                        panic!("cannot find value `{}` in this scope", name);
                    }
                }
            }
            Tag::Import => {}
            Tag::Type => {
                let name = self.tree.node_lexeme(id);
                let definition = self.lookup(name);
                match definition {
                    Definition::User(_) => {
                        self.definitions.insert(id, definition);
                    }
                    Definition::BuiltIn(_) => {
                        self.definitions.insert(id, definition);
                    }
                    Definition::Foreign(_) => {}
                    _ => {
                        panic!("cannot find value `{}` in this scope", name);
                    }
                }
                self.resolve_range(node)?;
            }
            Tag::VariableDecl => {
                let name = self.tree.node_lexeme(node.token);
                self.define_symbol(name, id)?;
                self.resolve_node(node.lhs)?;
                self.resolve_node(node.rhs)?;
            }
            _ => {
                self.resolve_node(node.lhs)?;
                self.resolve_node(node.rhs)?;
            }
        }
        Ok(())
    }

    /// Access: the lhs can be either another access or an identifier.
    fn get_container_definition(&self, container_id: NodeId) -> NodeId {
        match self.tree.node(container_id).tag {
            Tag::Access | Tag::Identifier => {
                // identifier -> variable decl / access -> field
                // If container_id is an Access,
                let type_def = self.get_definition_id(
                    container_id,
                    &format!(
                        "failed to find definition for identifier \"{:?}\".",
                        self.tree.node_lexeme(container_id)
                    ),
                );

                let def_node = self.tree.node(type_def);
                // variable decl -> struct decl / field -> struct decl
                let type_node_id = match def_node.tag {
                    Tag::Field => def_node.rhs,
                    _ => def_node.lhs,
                };

                let def_id = self.get_definition_id(
                    type_node_id,
                    &format!(
                        "failed to find struct definition for variable \"{}\".",
                        self.tree.node_lexeme(def_node.lhs)
                    ),
                );
                return def_id;
            }
            _ => {
                unreachable!("invalid container")
            }
        }
    }

    pub fn get_definition(&self, node_id: NodeId, msg: &str) -> &Definition {
        self.definitions.get(&node_id).expect(msg)
    }

    pub fn get_definition_id(&self, node_id: NodeId, msg: &str) -> NodeId {
        if let Definition::User(def_id) = self.get_definition(node_id, msg) {
            return *def_id;
        }
        unreachable!("{}", msg)
    }

    fn define_symbol(&mut self, name: &'a str, id: u32) -> Result<()> {
        let top = self.scopes.len() - 1;
        if let Err(_) = self.scopes[top].symbols.try_insert(name, id) {
            return Err(eyre!(" - Name \"{}\" is already defined.", name));
        }
        Ok(())
    }

    fn lookup(&self, name: &str) -> Definition {
        if self.scopes.len() == 0 {
            return Definition::NotFound;
        }
        let mut scope_index = self.current;
        loop {
            // Check if the name is defined in the current scope.
            if let Some(&index) = self.scopes[scope_index].get(name) {
                return Definition::User(index);
            }
            // Check if the name is defined in imported scopes.
            for &import_scope_index in &self.scopes[scope_index].unnamed_imports {
                if let Some(&index) = self.scopes[import_scope_index].get(name) {
                    return Definition::User(index);
                }
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
    }

    fn exit_scope(&mut self) {
        self.current = self.scopes[self.current].parent();
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
        for (i, scope) in self.scopes.iter().enumerate() {
            writeln!(f, "Scope [{}]: parent: {}", i, scope.parent);
            writeln!(f, "  {:?}", scope.symbols);
        }
        let mut references = HashMap::new();
        for (key, value) in &self.definitions {
            if let Err(mut error) = references.try_insert(value, vec![key]) {
                error.entry.get_mut().push(key);
            }
        }
        for (key, value) in &references {
            writeln!(
                f,
                "{:<12} x{:<3} {:?}",
                format!("{:?}", key),
                value.len(),
                value,
            );
        }
        Ok(())
    }
}
