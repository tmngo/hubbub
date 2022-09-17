use crate::{
    parse::{Node, NodeId, Tag, Tree},
    typecheck::TypeIndex,
    utils::assert_size,
};
use color_eyre::eyre::{eyre, Result, WrapErr};
use std::{collections::HashMap, fmt};

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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
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

#[derive(Debug, Clone)]
pub struct Scope<'a> {
    symbols: HashMap<&'a str, u32>,
    parent: usize,
    named_imports: Vec<usize>,
    unnamed_imports: Vec<usize>,
}

impl<'a> Scope<'a> {
    pub fn new(parent: usize) -> Self {
        Scope {
            symbols: HashMap::new(),
            parent,
            named_imports: Vec::new(),
            unnamed_imports: Vec::new(),
        }
    }

    pub fn from<const N: usize>(arr: [(&'a str, u32); N], parent: usize) -> Self {
        Self {
            symbols: HashMap::from(arr),
            parent,
            named_imports: Vec::new(),
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
                ("Array", TypeIndex::Array as u32),
                // ("alloc", 7),
                // ("print_int", 7),
            ],
            0,
        );
        let foreign = Scope::from(
            [
                // ("putchar", 1),
                // ("print_int", 2),
                // ("alloc", 3),
                // ("dealloc", 4),
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
        // Collect module declarations
        for i in root.lhs..root.rhs {
            let ni = self.tree.node_index(i);
            let module = self.tree.node(ni);
            if let Some(module_info) = self.tree.token_module(module.token) {
                let module_name = &module_info.name;
                dbg!(module_name);
            }
            println!("entering module scope");
            self.enter_scope();
            module_scopes.push(self.current);
            self.collect_module_decls(module)
                .wrap_err("Failed to collect module declarations")?;
            self.exit_scope();
        }
        // Copy symbols from unnamed imports
        for (i, scope) in self.scopes.clone().iter().enumerate() {
            self.current = i;
            for &import_scope_index in &scope.unnamed_imports.clone() {
                for (name, id) in self.scopes[import_scope_index].symbols.clone().iter() {
                    Self::define_symbol(&mut self.scopes[i], name, *id)?;
                }
            }
        }
        // Resolve module contents
        for (m, i) in (root.lhs..root.rhs).enumerate() {
            let ni = self.tree.node_index(i);
            let module = self.tree.node(ni);
            self.current = module_scopes[m];
            self.resolve_range(module)
                .wrap_err("Failed to resolve module definitions")?;
        }
        Ok(())
    }

    fn collect_module_decls(&mut self, module_node: &Node) -> Result<()> {
        for i in module_node.lhs..module_node.rhs {
            let node_id = self.tree.node_index(i);
            let node = self.tree.node(node_id);
            match node.tag {
                Tag::FunctionDecl => {
                    let name = self.tree.name(node_id);
                    let current_scope = &mut self.scopes[self.current];
                    Self::define_symbol(current_scope, name, node_id)?;
                }
                Tag::Import => {
                    // let module_name = self.tree.node_lexeme_offset(&node, 1).trim_matches('"');
                    // self.define_symbol(name, index)?;
                    // let module_info = self.tree.token_module(module_node.token);
                    let module_scope_index = self.get_module_scope_index(node);
                    let current_scope = &mut self.scopes[self.current];
                    if node.lhs != 0 {
                        // let alias_node = self.tree.node(node.lhs);
                        let module_alias = self.tree.name(node.lhs);
                        dbg!(module_alias);
                        current_scope.named_imports.push(module_scope_index);
                        // Map module_alias to Import node.
                        Self::define_symbol(current_scope, module_alias, node_id)?;
                    } else {
                        dbg!(module_scope_index);
                        // for (name, id) in self.scopes[module_scope_index].symbols.clone().iter()
                        // {
                        //     self.define_symbol(name, *id)?;
                        // }
                        current_scope.unnamed_imports.push(module_scope_index);
                    }
                }
                Tag::Struct => {
                    let name = self.tree.name(node_id);
                    let current_scope = &mut self.scopes[self.current];
                    Self::define_symbol(current_scope, name, node_id)?;
                    let mut scope = Scope::new(0);
                    // Collect field names
                    for i in node.lhs..node.rhs {
                        let field_id = self.tree.node_index(i);
                        let field_name = self.tree.name(field_id);
                        Self::define_symbol(&mut scope, field_name, field_id)?;
                    }
                    self.struct_scopes.insert(node_id, scope);
                }
                _ => {}
            };
        }
        Ok(())
    }

    fn get_module_scope_index(&self, module_node: &Node) -> usize {
        let module_name = self
            .tree
            .node_lexeme_offset(module_node, 1)
            .trim_matches('"');
        if let Some(module_index) = self.tree.get_module_index(module_name) {
            module_index + 2
        } else {
            unreachable!("failed to get module scope index")
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

    fn resolve_range(&mut self, node: &Node) -> Result<()> {
        if node.tag == Tag::Root {
            return Ok(());
        }
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
                let container_id = node.lhs;
                // rhs: member identifier
                // Assumes lhs is a declaration.
                self.resolve_node(container_id)?;
                // This should always return a struct definition.
                let container = self.tree.node(container_id);

                // Check if the container we're accessing is a module.
                match container.tag {
                    Tag::Access | Tag::Identifier => {
                        let container_def_id = self.definitions.get_definition_id(
                            container_id,
                            &format!(
                                "failed to find definition for container identifier \"{:?}\".",
                                self.tree.node_lexeme(container_id)
                            ),
                        );

                        let container_def = self.tree.node(container_def_id);
                        if let Tag::Import = container_def.tag {
                            // self.definitions.insert(id, definition);
                            let module_scope_index = self.get_module_scope_index(container_def);
                            // let member_definition = self.lookup(name: &str)
                            // self.definitions.insert(node.rhs, Definition::User())
                            let name = self.tree.node_lexeme(node.rhs);
                            // Check if the name is defined in the current scope.
                            let definition =
                                if let Some(&index) = self.scopes[module_scope_index].get(name) {
                                    Definition::User(index)
                                } else {
                                    Definition::NotFound
                                };
                            self.set_node_definition(id, name, definition.clone());
                            self.set_node_definition(node.rhs, name, definition);
                            return Ok(());
                        }
                    }
                    _ => {
                        unreachable!("invalid container")
                    }
                }

                // This doesn't need to be lookup() because structs are declared at the top level.
                let struct_decl_id = self.get_container_definition(container_id);
                // Resolve type
                if struct_decl_id != 0 {
                    let field_name = self.tree.name(node.rhs);

                    let field_id = match self
                        .struct_scopes
                        .get(&struct_decl_id)
                        .expect("failed to get struct scope")
                        .get(field_name)
                    {
                        Some(value) => value,
                        _ => {
                            return Err(eyre!(
                                "struct \"{}\" does not have member \"{}\".",
                                self.tree.name(struct_decl_id),
                                field_name
                            ));
                        }
                    };

                    // If this is a module access, map the access node to the module member.
                    // self.definitions.insert(
                    //     id, Definition::User(self.tree.node_index(index: u32))
                    // )
                    // Map access node to field_id.
                    self.definitions.insert(id, Definition::User(*field_id));
                    // Map member identifier to field_index.
                    let field = self.tree.node(*field_id);
                    let field_index = self.tree.node_index(field.rhs + 1);
                    self.definitions
                        .insert(node.rhs, Definition::User(field_index));
                }
                // else {
                //     return Err(eyre!("failed to look up lhs"));
                // }
            }
            Tag::Block | Tag::Module => {
                self.enter_scope();
                self.resolve_range(node)?;
                self.exit_scope();
            }
            Tag::Expressions | Tag::IfElse | Tag::Return | Tag::Struct => {
                self.resolve_range(node)?;
            }
            Tag::Field => {
                // Resolve type_expr
                self.resolve_node(self.tree.node_index(node.rhs))?;
            }
            Tag::FunctionDecl => {
                self.enter_scope();
                self.resolve_node(node.lhs)?; // Prototype
                if node.rhs != 0 {
                    let body = &self.tree.node(node.rhs);
                    self.resolve_range(body)?;
                }
                self.exit_scope();
            }
            Tag::Identifier => {
                let name = self.tree.node_lexeme(id);
                let definition = self.lookup(name);
                self.set_node_definition(id, name, definition);
            }
            Tag::Import => {}
            Tag::Parameters => {
                for i in node.lhs..node.rhs {
                    let field_id = self.tree.node_index(i);
                    let field = self.tree.node(field_id);
                    let name = self.tree.node_lexeme(field.lhs);
                    Self::define_symbol(&mut self.scopes[self.current], name, field_id)?;
                    // Resolve type_expr
                    self.resolve_node(field_id)?;
                }
            }
            Tag::Type => {
                let name = self.tree.node_lexeme(id);
                let definition = self.lookup(name);
                self.set_node_definition(id, name, definition);
                self.resolve_range(node)?;
            }
            Tag::TypeParameters => {
                for i in node.lhs..node.rhs {
                    let type_parameter_id = self.tree.node_index(i);
                    let name = self.tree.node_lexeme(type_parameter_id);
                    Self::define_symbol(&mut self.scopes[self.current], name, type_parameter_id)?;
                }
            }
            Tag::VariableDecl => {
                let name = self.tree.node_lexeme(node.token);
                Self::define_symbol(&mut self.scopes[self.current], name, id)?;
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

    fn set_node_definition(&mut self, node_id: NodeId, name: &str, definition: Definition) {
        match definition {
            Definition::User(_) => {
                self.definitions.insert(node_id, definition);
            }
            Definition::BuiltIn(_) => {
                self.definitions.insert(node_id, definition);
            }
            Definition::Foreign(_) => {}
            Definition::NotFound => {
                println!("{}", &self);
                panic!("cannot find `{}` in this scope", name);
            }
        }
    }

    /// Access: the lhs can be either another access or an identifier.
    fn get_container_definition(&self, container_id: NodeId) -> NodeId {
        match self.tree.node(container_id).tag {
            // The lhs of the Access is either an identfier, or another Access.
            Tag::Access | Tag::Identifier => {
                // identifier -> variable decl / access -> field
                // If container_id is an Access,
                let type_def = self.definitions.get_definition_id(
                    container_id,
                    &format!(
                        "failed to find definition for identifier \"{:?}\".",
                        self.tree.node_lexeme(container_id)
                    ),
                );

                let def_node = self.tree.node(type_def);
                // variable decl -> struct decl / field -> struct decl
                let type_node_id = match def_node.tag {
                    Tag::Field => self.tree.node_index(def_node.rhs),
                    _ => def_node.lhs,
                };

                if type_node_id == 0 {
                    return 0;
                }

                let def_id = self.definitions.get_definition_id(
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

    fn define_symbol(scope: &mut Scope<'a>, name: &'a str, id: u32) -> Result<()> {
        if scope.symbols.try_insert(name, id).is_err() {
            return Err(eyre!(" - Name \"{}\" is already defined.", name));
        }
        Ok(())
    }

    fn lookup(&self, name: &str) -> Definition {
        if self.scopes.is_empty() {
            return Definition::NotFound;
        }
        let mut scope_index = self.current;
        loop {
            // Check if the name is defined in the current scope.
            if let Some(&index) = self.scopes[scope_index].get(name) {
                return Definition::User(index);
            }
            // Check if the name is defined in imported scopes.
            // for &import_scope_index in &self.scopes[scope_index].unnamed_imports {
            //     if let Some(&index) = self.scopes[import_scope_index].get(name) {
            //         return Definition::User(index);
            //     }
            // }
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

pub trait Lookup {
    fn get_definition_id(&self, node_id: NodeId, msg: &str) -> NodeId;
}

impl Lookup for HashMap<NodeId, Definition> {
    fn get_definition_id(&self, node_id: NodeId, msg: &str) -> NodeId {
        let definition = self
            .get(&node_id)
            .unwrap_or_else(|| panic!("Definition not found: {}", msg));
        match definition {
            Definition::User(id) => *id,
            Definition::BuiltIn(id) => *id,
            Definition::Foreign(id) => *id,
            Definition::NotFound => unreachable!("Definition not found: {}", msg),
        }
    }
}

impl<'a> fmt::Display for Analyzer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, scope) in self.scopes.iter().enumerate() {
            writeln!(f, "Scope [{}]: parent: {}", i, scope.parent)?;
            writeln!(f, "  {:?}", scope.symbols)?;
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
            )?;
        }
        Ok(())
    }
}
