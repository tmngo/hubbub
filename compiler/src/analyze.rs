use crate::{
    parse::{Node, NodeId, Tag, Tree},
    typecheck::BuiltInType,
    utils::assert_size,
    workspace::{Result, Workspace},
};
use codespan_reporting::diagnostic::Diagnostic;
use std::{collections::HashMap, fmt, hash::Hash};

/**
 * This associates identifiers
 * Names are stored in a stack of maps.
 */
pub struct Analyzer<'a> {
    workspace: &'a mut Workspace,
    tree: &'a Tree,

    builtins: HashMap<&'a str, Definition>,
    current: usize,
    foreign: Scope<'a>,
    module_scopes: Vec<usize>,
    scopes: Vec<Scope<'a>>,
    struct_scopes: HashMap<u32, usize>,

    pub definitions: HashMap<NodeId, Definition>,
    pub overload_sets: HashMap<NodeId, Vec<Definition>>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Definition {
    BuiltIn(BuiltInType),
    BuiltInFunction(BuiltInFunction),
    User(NodeId),
    Foreign(u32),
    Overload(u32),
    Resolved(u32),
    NotFound,
}

impl Definition {
    pub fn id(&self) -> NodeId {
        match self {
            Definition::BuiltIn(id) => *id as NodeId,
            Definition::BuiltInFunction(id) => *id as NodeId,
            Definition::User(id)
            | Definition::Foreign(id)
            | Definition::Overload(id)
            | Definition::Resolved(id) => *id,
            Definition::NotFound => unreachable!("definition not found"),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum BuiltInFunction {
    Add,
    Mul,
    SizeOf,
}

// Assert that Tag size <= 1 byte
pub const _ASSERT_LOOKUP_SIZE: () = assert_size::<Definition>(8);

#[derive(Debug, Clone)]
pub struct Scope<'a> {
    symbols: HashMap<&'a str, u32>,
    fn_symbols: HashMap<&'a str, Vec<Definition>>,
    parent: usize,
    named_imports: Vec<usize>,
    unnamed_imports: Vec<usize>,
}

impl<'a> Scope<'a> {
    pub fn new(parent: usize) -> Self {
        Scope {
            symbols: HashMap::new(),
            fn_symbols: HashMap::new(),
            parent,
            named_imports: Vec::new(),
            unnamed_imports: Vec::new(),
        }
    }

    pub fn from<const N: usize>(arr: [(&'a str, u32); N], parent: usize) -> Self {
        Self {
            symbols: HashMap::from(arr),
            fn_symbols: HashMap::new(),
            parent,
            named_imports: Vec::new(),
            unnamed_imports: Vec::new(),
        }
    }

    pub fn get(&self, name: &str) -> Option<&u32> {
        self.symbols.get(name)
    }

    pub fn get_fn(&self, name: &str) -> Option<&Vec<Definition>> {
        self.fn_symbols.get(name)
    }

    pub fn parent(&self) -> usize {
        self.parent
    }
}

pub trait Resolve {
    fn define(&self, token: u32, lhs: u32, rhs: u32, analyzer: &mut Analyzer);
}

impl<'a> Analyzer<'a> {
    pub fn new(workspace: &'a mut Workspace, tree: &'a Tree) -> Self {
        let builtins = HashMap::from([
            ("Void", Definition::BuiltIn(BuiltInType::Void)),
            ("Bool", Definition::BuiltIn(BuiltInType::Boolean)),
            ("I64", Definition::BuiltIn(BuiltInType::Integer)),
            ("i64", Definition::BuiltIn(BuiltInType::Integer)),
            ("Int", Definition::BuiltIn(BuiltInType::Integer)),
            ("Int32", Definition::BuiltIn(BuiltInType::Integer)),
            ("Int64", Definition::BuiltIn(BuiltInType::Integer)),
            ("u8", Definition::BuiltIn(BuiltInType::Unsigned8)),
            ("Float", Definition::BuiltIn(BuiltInType::Float)),
            ("F64", Definition::BuiltIn(BuiltInType::Float)),
            // ("String", Definition::BuiltIn(BuiltInType::String)),
            ("Pointer", Definition::BuiltIn(BuiltInType::Pointer)),
            ("Array", Definition::BuiltIn(BuiltInType::Array)),
            ("+", Definition::BuiltInFunction(BuiltInFunction::Add)),
            ("*", Definition::BuiltInFunction(BuiltInFunction::Mul)),
            (
                "sizeof",
                Definition::BuiltInFunction(BuiltInFunction::SizeOf),
            ),
        ]);
        let foreign = Scope::from(
            [
                ("putchar", 1),
                ("puts", 1),
                // ("DisplayHelloFromMyDLL", 9),
                // ("print_int", 2),
                // ("alloc", 3),
                // ("dealloc", 4),
            ],
            0,
        );
        Self {
            workspace,
            tree,
            builtins,
            current: 0,
            foreign,
            module_scopes: vec![],
            scopes: Vec::new(),
            struct_scopes: HashMap::new(),
            definitions: HashMap::new(),
            overload_sets: HashMap::new(),
        }
    }

    pub fn resolve(&mut self) -> Result<()> {
        self.enter_scope();
        let root = &self.tree.node(0);
        // Collect module declarations
        for _ in root.lhs..root.rhs {
            self.enter_scope();
            self.module_scopes.push(self.current);
            self.exit_scope();
        }
        for i in root.lhs..root.rhs {
            let ni = self.tree.node_index(i);
            let module = self.tree.node(ni);
            self.current = self.module_scopes[(i - root.lhs) as usize];
            self.collect_module_decls(module)?;
        }
        // Copy symbols from unnamed imports
        for (i, scope) in self.scopes.clone().iter().enumerate() {
            for &import_scope_index in &scope.unnamed_imports.clone() {
                for (name, id) in self.scopes[import_scope_index].symbols.clone().iter() {
                    Self::define_symbol(self.tree, &mut self.scopes[i], name, *id)?;
                }
                for (name, ids) in self.scopes[import_scope_index].fn_symbols.clone().iter() {
                    Self::define_function(&mut self.scopes[i], name, ids)?;
                }
            }
        }
        // Resolve module contents
        for (m, i) in (root.lhs..root.rhs).enumerate() {
            let ni = self.tree.node_index(i);
            let module = self.tree.node(ni);
            self.current = self.module_scopes[m];
            self.resolve_range(module)?;
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
                    // dbg!(self.current);
                    // dbg!(name);
                    Self::define_function(current_scope, name, &[Definition::User(node_id)])?;
                }
                Tag::Import => {
                    // let module_name = self.tree.node_lexeme_offset(&node, 1).trim_matches('"');
                    // self.define_symbol(name, index)?;
                    // let module_info = self.tree.token_module(module_node.token);
                    let module_scope_index = self.get_module_scope_index(node);
                    let current_scope = &mut self.scopes[self.current];
                    if node.lhs != 0 {
                        let module_alias = self.tree.name(node.lhs);
                        // dbg!(module_alias);
                        current_scope.named_imports.push(module_scope_index);
                        // Map module_alias to Import node.
                        Self::define_symbol(self.tree, current_scope, module_alias, node_id)?;
                    } else {
                        // dbg!(module_scope_index);
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
                    Self::define_symbol(self.tree, current_scope, name, node_id)?;
                    let mut scope = Scope::new(self.current);
                    // Define type parameters
                    if node.lhs != 0 {
                        let type_parameters = self.tree.node(node.lhs);
                        for i in self.tree.range(type_parameters) {
                            let id = self.tree.node_index(i);
                            let name = self.tree.node_lexeme(id);
                            Self::define_symbol(self.tree, &mut scope, name, id)?;
                        }
                    }
                    // Collect field names
                    for i in self.tree.range(node) {
                        let field_id = self.tree.node_index(i);
                        let field_name = self.tree.name(field_id);
                        Self::define_symbol(self.tree, &mut scope, field_name, field_id)?;
                    }
                    self.scopes.push(scope.clone());
                    self.struct_scopes.insert(node_id, self.scopes.len() - 1);
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
            self.module_scopes[module_index]
        } else {
            unreachable!("failed to get module scope index")
        }
    }

    fn resolve_range(&mut self, node: &Node) -> Result<()> {
        if node.tag == Tag::Root {
            return Ok(());
        }
        for i in self.tree.range(node) {
            let ni = self.tree.node_index(i);
            let result = self.resolve_node(ni);
            if let Err(diagnostic) = result {
                self.workspace.diagnostics.push(diagnostic);
            }
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

                // The lhs of the Access is either an identfier, or another Access.
                // identifier -> variable decl
                // access -> field
                let container_decl_id = self.definitions.get_definition_id(
                    container_id,
                    &format!(
                        "failed to find definition for identifier \"{:?}\".",
                        self.tree.node_lexeme(container_id)
                    ),
                );
                // Check if the container we're accessing is a module.
                match container.tag {
                    Tag::Access | Tag::Identifier => {
                        let container_def = self.tree.node(container_decl_id);
                        if let Tag::Import = container_def.tag {
                            // self.definitions.insert(id, definition);
                            let module_scope_index = self.get_module_scope_index(container_def);
                            // let member_definition = self.lookup(name: &str)
                            // self.definitions.insert(node.rhs, Definition::User())
                            let name = self.tree.node_lexeme(node.rhs);
                            // Check if the name is defined in the current scope.
                            let definition =
                                self.lookup_in_scope(module_scope_index, node.rhs, name);
                            if let Definition::NotFound = definition {
                                let module_name = if node.lhs == 0 {
                                    self.tree.name(node.lhs)
                                } else {
                                    self.tree
                                        .node_lexeme_offset(container_def, 1)
                                        .trim_matches('"')
                                };
                                let name = self.tree.name(node.rhs);
                                let token_id = self.tree.node(node.rhs).token;
                                return Err(Diagnostic::error()
                                    .with_message(format!(
                                        "cannot find \"{}\" in module \"{}\".",
                                        name, module_name
                                    ))
                                    .with_labels(vec![self.tree.label(token_id)]));
                            }
                            self.set_node_definition(id, definition)?;
                            self.set_node_definition(node.rhs, definition)?;
                            return Ok(());
                        }
                    }
                    _ => {
                        unreachable!("invalid container")
                    }
                }

                // This doesn't need to be lookup() because structs are declared at the top level.
                let struct_decl_id = self.get_container_definition(container_decl_id);
                // Resolve type
                if struct_decl_id != 0 {
                    let field_name = self.tree.name(node.rhs);
                    let struct_scope_index = self
                        .struct_scopes
                        .get(&struct_decl_id)
                        .expect("failed to get struct scope");
                    let struct_scope = &self.scopes[*struct_scope_index];
                    let field_id = match struct_scope.get(field_name) {
                        Some(value) => value,
                        _ => {
                            let token_id = self.tree.node(node.rhs).token;
                            return Err(Diagnostic::error()
                                .with_message(format!(
                                    "no field \"{}\" on type \"{}\".",
                                    field_name,
                                    self.tree.name(struct_decl_id)
                                ))
                                .with_labels(vec![self.tree.label(token_id)]));
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
            }
            Tag::Add | Tag::Mul => {
                let definition = self.lookup(id);
                self.set_node_definition(id, definition)?;
                self.resolve_node(node.lhs)?;
                self.resolve_node(node.rhs)?;
            }
            Tag::Block | Tag::Module => {
                self.enter_scope();
                self.resolve_range(node)?;
                self.exit_scope();
            }
            Tag::Expressions | Tag::IfElse | Tag::Return => {
                self.resolve_range(node)?;
            }
            Tag::Field => {
                // Resolve type_expr
                self.resolve_node(self.tree.node_extra(node, 0))?;
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
                let definition = self.lookup(id);
                self.set_node_definition(id, definition)?;
            }
            Tag::Import => {}
            Tag::Parameters => {
                for i in node.lhs..node.rhs {
                    let field_id = self.tree.node_index(i);
                    let field = self.tree.node(field_id);
                    let name = self.tree.node_lexeme(field.lhs);
                    Self::define_symbol(self.tree, &mut self.scopes[self.current], name, field_id)?;
                    // Resolve type_expr
                    self.resolve_node(field_id)?;
                }
            }
            Tag::Prototype => {
                self.resolve_node(self.tree.node_extra(node, 0))?;
                self.resolve_node(self.tree.node_extra(node, 1))?;
            }
            Tag::Struct => {
                let current_scope = self.current;
                self.current = *self
                    .struct_scopes
                    .get(&id)
                    .expect("failed to get struct scope");
                self.resolve_range(node)?;
                self.current = current_scope;
            }
            Tag::Type => {
                let definition = self.lookup(id);
                self.set_node_definition(id, definition)?;
                self.resolve_range(node)?;
            }
            Tag::TypeParameters => {
                for i in node.lhs..node.rhs {
                    let type_parameter_id = self.tree.node_index(i);
                    let name = self.tree.node_lexeme(type_parameter_id);
                    Self::define_symbol(
                        self.tree,
                        &mut self.scopes[self.current],
                        name,
                        type_parameter_id,
                    )?;
                }
            }
            Tag::VariableDecl => {
                // Resolve identifier(s).
                let lhs = self.tree.node(node.lhs);
                match lhs.tag {
                    Tag::Expressions => {
                        for i in lhs.lhs..lhs.rhs {
                            let ni = self.tree.node_index(i);
                            let name = self.tree.name(ni);
                            Self::define_symbol(
                                self.tree,
                                &mut self.scopes[self.current],
                                name,
                                ni,
                            )?;
                        }
                    }
                    Tag::Identifier => {
                        let ni = node.lhs;
                        let name = self.tree.name(ni);
                        Self::define_symbol(self.tree, &mut self.scopes[self.current], name, ni)?;
                    }
                    _ => {}
                }
                self.resolve_node(self.tree.node_extra(node, 0))?;
                self.resolve_node(self.tree.node_extra(node, 1))?;
            }
            _ => {
                self.resolve_node(node.lhs)?;
                self.resolve_node(node.rhs)?;
            }
        }
        Ok(())
    }

    fn set_node_definition(&mut self, node_id: NodeId, definition: Definition) -> Result<()> {
        match definition {
            Definition::User(_)
            | Definition::Overload(_)
            | Definition::BuiltIn(_)
            | Definition::BuiltInFunction(_)
            | Definition::Resolved(_) => {
                self.definitions.insert(node_id, definition);
            }
            Definition::Foreign(_) => {
                self.definitions.insert(node_id, definition);
            }
            Definition::NotFound => {
                let name = self.tree.name(node_id);
                let token_id = self.tree.node(node_id).token;
                return Err(Diagnostic::error()
                    .with_message(format!("cannot find \"{}\" in this scope.", name))
                    .with_labels(vec![self.tree.label(token_id)]));
            }
        }
        Ok(())
    }

    /// Access: the lhs can be either another access or an identifier.
    fn get_container_definition(&self, container_decl_id: NodeId) -> NodeId {
        let def_node = self.tree.node(container_decl_id);
        let type_node_id = match def_node.tag {
            // field -> struct decl
            Tag::Field => self.tree.node_index(def_node.rhs),
            // variable decl -> struct decl
            Tag::VariableDecl => self.tree.node_extra(def_node, 0),
            // identifier -> var decl
            Tag::Identifier => {
                let variable_decl = self.tree.node(def_node.lhs);
                self.tree.node_extra(variable_decl, 0)
            }
            _ => unreachable!(),
        };

        if type_node_id == 0 {
            return 0;
        }

        self.definitions.get_definition_id(
            type_node_id,
            &format!(
                "failed to find struct definition for variable \"{}\".",
                self.tree.node_lexeme(def_node.lhs)
            ),
        )
    }

    fn define_symbol(tree: &Tree, scope: &mut Scope<'a>, name: &'a str, id: u32) -> Result<()> {
        if let Err(err) = scope.symbols.try_insert(name, id) {
            let node = tree.node(id);
            let previous = tree.node(*err.entry.get());
            return Err(Diagnostic::error()
                .with_message(format!("The name \"{}\" is already defined.", name))
                .with_labels(vec![
                    tree.label(node.token),
                    tree.label(previous.token)
                        .with_message("previous definition"),
                ]));
        }
        Ok(())
    }

    fn define_function(scope: &mut Scope<'a>, name: &'a str, ids: &[Definition]) -> Result<()> {
        if let Err(mut error) = scope.fn_symbols.try_insert(name, Vec::from(ids)) {
            error.entry.get_mut().extend(ids);
        }
        Ok(())
    }

    fn lookup(&mut self, id: NodeId) -> Definition {
        let name = self.tree.name(id);
        if self.scopes.is_empty() {
            return Definition::NotFound;
        }
        let mut scope_index = self.current;
        let mut overload_set = Vec::<Definition>::new();
        loop {
            // Check if the name is defined in the current scope.
            if let Some(&index) = self.scopes[scope_index].get(name) {
                return Definition::User(index);
            }
            if let Some(definitions) = self.scopes[scope_index].get_fn(name) {
                overload_set.extend(definitions);
            }
            // Check if the name is defined in imported scopes.
            // for &import_scope_index in &self.scopes[scope_index].unnamed_imports {
            //     if let Some(&index) = self.scopes[import_scope_index].get(name) {
            //         return Definition::User(index);
            //     }
            // }
            // If the name isn't in the global scope, it's undefined.
            if scope_index == 0 {
                if let Some(&definition) = self.builtins.get(name) {
                    match definition {
                        Definition::BuiltIn(_) => return definition,
                        Definition::BuiltInFunction(_) => overload_set.push(definition),
                        _ => unreachable!(),
                    }
                }
                if let Some(&index) = self.foreign.get(name) {
                    return Definition::Foreign(index);
                }
                return match overload_set.len() {
                    0 => Definition::NotFound,
                    1 => overload_set[0],
                    _ => {
                        self.overload_sets.insert(id, overload_set);
                        Definition::Overload(id)
                    }
                };
            }
            // Look in the parent scope.
            scope_index = self.scopes[scope_index].parent();
        }
    }

    fn lookup_in_scope(&mut self, scope_index: usize, id: NodeId, name: &str) -> Definition {
        if self.scopes.is_empty() {
            return Definition::NotFound;
        }
        // Check if the name is defined in the current scope.
        if let Some(&index) = self.scopes[scope_index].get(name) {
            return Definition::User(index);
        }
        if let Some(ids) = self.scopes[scope_index].get_fn(name) {
            let overload_set = ids.clone();
            return match ids.len() {
                0 => Definition::NotFound,
                1 => Definition::User(overload_set[0].id()),
                _ => {
                    self.overload_sets.insert(id, overload_set);
                    Definition::Overload(id)
                }
            };
        }
        // If the name isn't in the global scope, it's undefined.
        if scope_index == 0 {
            if let Some(&definition) = self.builtins.get(name) {
                return definition;
            }
            if let Some(&index) = self.foreign.get(name) {
                return Definition::Foreign(index);
            }
        }
        Definition::NotFound
    }

    fn enter_scope(&mut self) {
        let parent = self.current;
        self.scopes.push(Scope::new(parent));
        self.current = self.scopes.len() - 1;
    }

    fn exit_scope(&mut self) {
        self.current = self.scopes[self.current].parent();
    }
}

pub trait Lookup {
    fn get_definition_id(&self, node_id: NodeId, msg: &str) -> NodeId;
}

impl Lookup for HashMap<NodeId, Definition> {
    fn get_definition_id(&self, node_id: NodeId, msg: &str) -> NodeId {
        let definition = self
            .get(&node_id)
            .unwrap_or_else(|| panic!("Definition not found: {}", msg));
        definition.id()
    }
}

impl<'a> fmt::Display for Analyzer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, scope) in self.scopes.iter().enumerate() {
            writeln!(f, "Scope [{}]: parent: {}", i, scope.parent)?;
            writeln!(f, "  {:?} {:?}", scope.symbols, scope.fn_symbols)?;
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
        for (i, set) in &self.overload_sets {
            writeln!(f, "Set [{}]: {:?}", i, set)?;
        }
        Ok(())
    }
}
