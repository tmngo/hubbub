use crate::{
    tokenize::{Tag as TokenTag, Token, Tokenizer},
    utils::assert_size,
    workspace::{Result, Workspace},
};
use codespan_reporting::diagnostic::{Diagnostic, Label};
use std::{
    fmt::{self, Debug},
    ops::Range,
    path::PathBuf,
};

/*
program = decl*


DECLARATIONS

    decl            = var-decl | func-decl | struct-decl

    var-decl        = identifier ':=' expr
                    | identifier ':' type
                    | identifier ':' type '=' expr
                    | identifier '::' expr

    func-decl       = name '::' '(' parameter-list ')' stmt* 'end'

        prototype   = '(' parameter-list ')' '->' identifier
                    | '(' parameter-list ')' '->' '(' expr-list ')'

    struct-decl     = name '::' 'struct' field* 'end'


STATEMENTS

    stmt            = var-decl
                    | assign-stmt
                    | return-stmt | continue-stmt | break-stmt

    assign-stmt     = expr (assign-op expr)?

    stmt-if         = 'if' expr stmt* ('elseif' expr stmt*) ('else' stmt*)? 'end'

    stmt-while      = 'while' expr stmt* 'end'


EXPRESSIONS

    expr            = expr-prefix (binary-op expr-prefix)?

    expr-prefix     = prefix-op* expr-postfix

        prefix-op   = '!' | '-' | '~'

    expr-postfix    = expr-operand postfix-op*

        postfix-op  = '!' | '?'
                    | '.' identifier
                    | '[' expr ']'
                    | '(' expr-list ')'

    expr-operand    = expr-if | expr-base

    expr-base       = expr-type | expr-group | literal

    expr-type       = identifier
                    | identifier '{' expr-list '}'
                    | prototype

    expr-group      = '(' expr ')'


BLOCKS

    expr-block      = 'block' stmt* 'end'

    expr-if         = 'if' expr stmt* ('else if' expr stmt* )* ('else' stmt*)? 'end'

    expr-while      = 'while' expr stmt* 'end'


LISTS

    parameter-list  = (parameter ',')* parameter?

    expr-list       = (expr ',')* expr?

*/

// Assert that Tag size <= 1 byte
pub const _ASSERT_TAG_SIZE: () = assert_size::<Tag>(1);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Tag {
    Access,              // lhs, rhs
    Add,                 // lhs, rhs
    Address,             // expr
    LogicalAnd,          // lhs, rhs
    LogicalOr,           // lhs, rhs
    Assign,              // lhs, rhs
    AssignAdd,           // lhs, rhs
    Block,               // start..end [Stmt]
    BlockDirect,         //
    BitwiseAnd,          // expr
    BitwiseNot,          // expr
    BitwiseOr,           // lhs, rhs
    BitwiseShiftL,       //
    BitwiseShiftR,       //
    BitwiseXor,          // lhs, rhs
    Break,               //
    Call,                // func_expr, arguments: Expressions
    Continue,            //
    Equality,            // lhs, rhs
    Expressions,         // start..end [Expr]
    Dereference,         // expr
    Factorial,           // expr
    False,               //
    Field,               // type_expr
    FunctionDecl,        // prototype, block
    Greater,             // lhs, rhs
    Grouping,            // expr
    Identifier,          //
    If,                  // condition, block
    IfElse,              // start..end [If]
    Import,              // lhs, rhs
    Inequality,          // lhs, rhs
    IntegerLiteral,      //
    Invalid,             //
    Less,                // lhs, rhs
    Div,                 // lhs, rhs
    Module,              // start..end [Declaration]
    Mul,                 // lhs, rhs
    Negation,            // expr
    Not,                 // expr
    Parameters,          // start..end [Field]
    ParametricPrototype, //
    Prototype,           // parameters: Parameters, returns: Expressions
    Return,              // expr
    Root,                // start..end [Decl]
    StringLiteral,       //
    Struct,              // start..end [Field]
    Sub,                 // lhs, rhs
    Subscript,           // lhs, rhs
    True,                //
    Type,                // expr
    TypeParameter,       // start..end [Identifier]
    TypeParameters,      // start..end [Identifier]
    VariableDecl,        // type_expr, init_expr
    While,               // condition, block
}

// pub struct Node2 {
//     token: u32,
//     data: Data,
// }

// pub enum Data {
//     Add { lhs: u32, rhs: u32 },
//     And { lhs: u32, rhs: u32 },
//     Assign { lhs: u32, rhs: u32 },
//     Block { start: u32, end: u32 },
//     BlockDirect,
//     BitwiseAnd { lhs: u32, rhs: u32 },
//     BitwiseNot { expr: u32 },
//     BitwiseOr { lhs: u32, rhs: u32 },
//     BitwiseXor { lhs: u32, rhs: u32 },
//     Break,
//     Call { function: u32, arguments: u32 },
//     Continue,
//     Div { lhs: u32, rhs: u32 },
//     Expressions { start: u32, end: u32 },
//     Field { type_expr: u32 },
//     FunctionDecl { prototype: u32, block: u32 },
//     Greater { lhs: u32, rhs: u32 },
//     Grouping { expr: u32 },
//     Identifier,
//     If { condition: u32, block: u32 },
//     IfElse { start: u32, end: u32 },
//     IntegerLiteral,
//     Invalid,
//     Less { lhs: u32, rhs: u32 },
//     Mul { lhs: u32, rhs: u32 },
//     Negation { expr: u32 },
//     Not { expr: u32 },
//     Parameters { start: u32, end: u32 },
//     Prototype { parameters: u32, returns: u32 },
//     Return { expr: u32 },
//     Root { start: u32, end: u32 },
//     StringLiteral,
//     Struct { start: u32, end: u32 },
//     Sub { lhs: u32, rhs: u32 },
//     TypeBase { expr: u32 },
//     VariableDecl { type_expr: u32, init_expr: u32 },
//     While { condition: u32, block: u32 },
// }

// Assert that Node size <= 16 bytes
pub const _ASSERT_NODE_SIZE: () = assert_size::<Node>(16);
// pub const _ASSERT_NODE_2_SIZE: () = assert_size::<Node2>(16);
// pub const _ASSERT_NODE_DATA_SIZE: () = assert_size::<Data>(12);

type TokenId = u32;
pub type NodeId = u32;
// pub type Id = u32;

#[derive(Copy, Clone, Debug)]
pub struct Node1 {
    pub tag: Tag,
    pub token: TokenId,
    pub lhs: u32,
    pub rhs: u32,
}

pub struct NodeIterator {
    index: u32,
    end: u32,
}

impl IntoIterator for &Node1 {
    type Item = u32;
    type IntoIter = NodeIterator;
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            index: self.lhs,
            end: self.rhs,
        }
    }
}

impl Iterator for NodeIterator {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        let current = self.index;
        self.index += 1;
        if current < self.end {
            Some(current)
        } else {
            None
        }
    }
}

pub type Node = Node1;

#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub alias: Option<String>,
    pub first_token_id: u32,
}

#[derive(Copy, Clone)]
enum Associativity {
    Left,
    None,
    Right,
}

#[derive(Copy, Clone)]
struct Operator {
    tag: Tag,
    precedence: i32,
    // left_precedence: i8,
    // right_precedence: i8,
    associativity: Associativity,
}

impl Operator {
    fn new(tag: Tag, precedence: i32, associativity: Associativity) -> Self {
        Operator {
            tag,
            precedence,
            associativity,
        }
    }
}

pub struct Parser<'w> {
    /// Current token index.
    workspace: &'w mut Workspace,
    index: usize,
    /// Stores node indices temporarily.
    stack: Vec<u32>,
    /// Stores AST nodes.
    pub tree: Tree,
}

fn token_to_operator(tag: TokenTag) -> Operator {
    match tag {
        TokenTag::PipePipe => Operator::new(Tag::LogicalOr, 1, Associativity::Left),
        //
        TokenTag::AmpersandAmpersand => Operator::new(Tag::LogicalAnd, 2, Associativity::Left),
        //
        TokenTag::BangEqual => Operator::new(Tag::Inequality, 3, Associativity::None),
        TokenTag::EqualEqual => Operator::new(Tag::Equality, 3, Associativity::None),
        TokenTag::Greater => Operator::new(Tag::Greater, 3, Associativity::None),
        TokenTag::Less => Operator::new(Tag::Less, 3, Associativity::None),
        //
        TokenTag::Ampersand => Operator::new(Tag::BitwiseAnd, 4, Associativity::Left),
        TokenTag::Caret => Operator::new(Tag::BitwiseXor, 4, Associativity::Left),
        TokenTag::Pipe => Operator::new(Tag::BitwiseOr, 4, Associativity::Left),
        TokenTag::LessLess => Operator::new(Tag::BitwiseShiftL, 4, Associativity::Left),
        TokenTag::GreaterGreater => Operator::new(Tag::BitwiseShiftR, 4, Associativity::Left),
        //
        TokenTag::Minus => Operator::new(Tag::Sub, 6, Associativity::Left),
        TokenTag::Plus => Operator::new(Tag::Add, 6, Associativity::Left),
        //
        TokenTag::Slash => Operator::new(Tag::Div, 7, Associativity::Left),
        TokenTag::Star => Operator::new(Tag::Mul, 7, Associativity::Left),
        //
        _ => Operator::new(Tag::Invalid, -1, Associativity::None),
    }
}

/// The body should return a NodeId.
macro_rules! parse_while {
    ($self:expr, $cond:expr, $body:block) => {{
        let stack_top = $self.stack.len();
        while !$self.is_at_end() && $cond {
            let node: NodeId = $body;
            $self.stack.push(node);
        }
        $self.add_indices(stack_top)
    }};
}

impl<'w> Parser<'w> {
    pub fn new(workspace: &'w mut Workspace, source: &str, tokens: Vec<Token>) -> Self {
        Self {
            workspace,
            stack: Vec::new(),
            index: 0,
            tree: Tree {
                sources: vec![source.to_string()],
                tokens,
                nodes: vec![Node {
                    tag: Tag::Root,
                    token: 0,
                    lhs: 0,
                    rhs: 0,
                }],
                indices: Vec::new(),
                modules: Vec::from([Module {
                    name: String::new(),
                    alias: None,
                    first_token_id: 0,
                }]),
            },
        }
    }

    pub fn parse(&mut self) {
        let result = self.parse_modules();
        if let Err(diagnostic) = result {
            self.workspace.diagnostics.push(diagnostic);
        }
    }

    pub fn tree(self) -> Tree {
        self.tree
    }

    /**************************************************************************/
    // Declarations

    /// program = module*
    fn parse_modules(&mut self) -> Result<Range<NodeId>> {
        let range = parse_while!(self, true, { self.parse_module()? });
        self.tree.nodes[0] = Node {
            tag: Tag::Root,
            token: 0,
            lhs: range.start,
            rhs: range.end,
        };
        Ok(range)
    }

    /// module = root-decl*
    fn parse_module(&mut self) -> Result<NodeId> {
        let first_module_token = self.index as TokenId;
        self.match_token(TokenTag::Newline);
        let range = parse_while!(self, true, { self.parse_root_declaration()? });
        self.expect_token(TokenTag::Eof)?;
        self.add_node(Tag::Module, first_module_token, range.start, range.end)
    }

    /// root-decl = fn-decl | var-decl | struct-decl | module-decl
    fn parse_root_declaration(&mut self) -> Result<NodeId> {
        while self.match_token(TokenTag::Newline) {}
        // Look at the token after the '::' or ':'.
        if self.current_tag() == TokenTag::ForeignLibrary {
            self.parse_foreign_library()?;
            return self.parse_root_declaration();
        }
        if self.current_tag() == TokenTag::Import {
            return self.parse_module_import();
        }
        match self.next_tag(2) {
            TokenTag::BraceL => self.parse_decl_function(),
            TokenTag::ParenL => {
                // identifier :: ()
                if self.next_tag(3) == TokenTag::ParenR {
                    return self.parse_decl_function();
                }
                // identifier :: (parameter: ...
                if self.next_tag(3) == TokenTag::Identifier && self.next_tag(4) == TokenTag::Colon {
                    return self.parse_decl_function();
                }
                // identifier :: ( ...
                self.parse_decl_variable()
            }
            // TokenTag::Function => return self.parse_decl_function(),
            // identifier :: struct ...
            TokenTag::Import => self.parse_module_import(),
            TokenTag::Operator => self.parse_decl_function(),
            TokenTag::Struct => self.parse_decl_struct(),
            // identifier :: ...
            _ => self.parse_decl_variable(),
        }
    }

    /// module-import = identifier '::' 'module' string-literal
    fn parse_module_import(&mut self) -> Result<NodeId> {
        let (lhs, alias) = if self.current_tag() == TokenTag::Identifier {
            let alias_token = self.expect_token(TokenTag::Identifier)?;
            self.expect_token(TokenTag::ColonColon)?;
            let lhs = self.add_leaf(Tag::Identifier, alias_token)?;
            (lhs, Some(self.tree.node_lexeme(lhs).to_string()))
        } else {
            (0, None)
        };

        let module_token = self.expect_token(TokenTag::Import)?;
        let name_token = self.expect_token(TokenTag::StringLiteral)?;
        let rhs = self.add_leaf(Tag::StringLiteral, name_token)?;
        let current_source = self.tree.token_source(name_token).0;
        let module_name = self
            .tree
            .token(name_token)
            .to_str(current_source)
            .trim_matches('"')
            .to_string();

        if self.tree.get_module_index(&module_name).is_none() {
            let filename = &format!("{}.hb", module_name);
            let path = if let Ok(value) = std::env::var("ABSOLUTE_MODULE_PATH") {
                PathBuf::from(value).join(filename)
            } else {
                std::env::current_exe()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .join("modules")
                    .join(filename)
            };
            if path.exists() {
                self.tree.modules.push(Module {
                    name: module_name,
                    alias,
                    first_token_id: self.tree.tokens.len() as TokenId,
                });
                let source = std::fs::read_to_string(&path).unwrap();
                let mut tokenizer = Tokenizer::new(&source);
                tokenizer.append_tokens(&mut self.tree.tokens);
                self.tree.sources.push(source.clone());
                self.workspace.files.add(
                    path.file_name().unwrap().to_owned().into_string().unwrap(),
                    source,
                );
            } else {
                return Err(Diagnostic::error().with_message(format!(
                    "Failed to find module \"{}\" at path {:?}",
                    module_name, path
                )));
            }
        }
        self.match_token(TokenTag::Newline);
        self.add_node(Tag::Import, module_token, lhs, rhs)
    }

    fn parse_foreign_library(&mut self) -> Result<()> {
        self.expect_token(TokenTag::ForeignLibrary)?;
        let name_token = self.expect_token(TokenTag::StringLiteral)?;
        let current_source = self.tree.token_source(name_token).0;
        let library_name = self
            .tree
            .token(name_token)
            .to_str(current_source)
            .trim_matches('"')
            .to_string();
        for name in &self.workspace.library_files {
            if &library_name == name {
                return Ok(());
            }
        }
        self.workspace.library_files.push(library_name);
        Ok(())
    }

    /// function-decl = ideentifier '::' '(' ')'
    fn parse_decl_function(&mut self) -> Result<NodeId> {
        self.shift_token(); // Identifier or operator
        let token_index = self.expect_token(TokenTag::ColonColon)?; // '::'
        self.match_token(TokenTag::Operator);
        let prototype = self.parse_parametric_prototype()?;
        let body = if self.current_tag() == TokenTag::End {
            self.expect_token_and_newline(TokenTag::End)?; // 'end'
            0
        } else {
            self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon])?;
            self.parse_stmt_body(0)?
        };
        self.add_node(Tag::FunctionDecl, token_index, prototype, body)
    }

    fn parse_parametric_prototype(&mut self) -> Result<NodeId> {
        let token = self.index as TokenId;
        if self.current_tag() == TokenTag::BraceL {
            let type_parameters = self.parse_type_parameters()?;
            let prototype = self.parse_prototype()?;
            self.add_node(Tag::ParametricPrototype, token, type_parameters, prototype)
        } else {
            self.parse_prototype()
        }
    }

    /// type-parameters = '{' type-parameter (',' type-parameter)* ','? '}'
    fn parse_type_parameters(&mut self) -> Result<NodeId> {
        let token = self.expect_token(TokenTag::BraceL)?;
        let range = parse_while!(self, self.token_isnt(TokenTag::BraceR), {
            let identifier_token = self.expect_token(TokenTag::Identifier)?;
            self.match_token(TokenTag::Comma);
            self.add_leaf(Tag::TypeParameter, identifier_token)?
        });
        self.expect_token(TokenTag::BraceR)?;
        self.add_node(Tag::TypeParameters, token, range.start, range.end)
    }

    /// prototype =
    fn parse_prototype(&mut self) -> Result<NodeId> {
        let parameters = self.parse_parameters()?;
        let returns = if self.match_token(TokenTag::Arrow) {
            self.parse_type_list()?
        } else {
            0
        };
        self.add_node(Tag::Prototype, 0, parameters, returns)
    }

    /// parameters = '(' field (',' field)* ')'
    fn parse_parameters(&mut self) -> Result<NodeId> {
        let token = self.expect_token(TokenTag::ParenL)?;
        let mut source_index = 0;
        let range = parse_while!(self, self.token_isnt(TokenTag::ParenR), {
            let field = self.parse_field(source_index)?;
            source_index += 1;
            self.match_token(TokenTag::Comma);
            field
        });
        self.expect_token(TokenTag::ParenR)?;
        self.add_node(Tag::Parameters, token, range.start, range.end)
    }

    /// type-list = expr-base | '(' (expr-base ',')* ')'
    fn parse_type_list(&mut self) -> Result<NodeId> {
        if self.match_token(TokenTag::ParenL) {
            let range = parse_while!(self, self.token_isnt(TokenTag::ParenR), {
                let type_expr = self.parse_expr_base()?;
                self.match_token(TokenTag::Comma);
                type_expr
            });
            self.expect_token(TokenTag::ParenR)?;
            self.add_node(Tag::Expressions, 0, range.start, range.end)
        } else {
            self.parse_expr_base()
        }
    }

    /// identifier-list = identifier (',' identifier)* ','?
    fn parse_identifier_list(&mut self) -> Result<NodeId> {
        if self.next_tag(1) == TokenTag::Comma && self.next_tag(2) == TokenTag::Identifier {
            let range = parse_while!(
                self,
                self.token_isnt(TokenTag::Colon) && !self.token_is(TokenTag::ColonEqual),
                {
                    let identifier_token = self.expect_token(TokenTag::Identifier)?;
                    self.match_token(TokenTag::Comma);
                    self.add_leaf(Tag::Identifier, identifier_token)?
                }
            );
            let first_identifer = self.tree.node(self.tree.node_index(range.start));
            self.add_node(
                Tag::Expressions,
                first_identifer.token,
                range.start,
                range.end,
            )
        } else {
            let identifier_token = self.expect_token(TokenTag::Identifier)?;
            self.match_token(TokenTag::Comma);
            self.add_leaf(Tag::Identifier, identifier_token)
        }
    }

    /// expr-list = expr (',' expr)* ','?
    fn parse_expr_list(&mut self) -> Result<NodeId> {
        let range = parse_while!(
            self,
            self.token_isnt(TokenTag::Newline) && self.token_isnt(TokenTag::Semicolon),
            {
                let expr = self.parse_expr()?;
                // println!("{:?}", format!("{}", TreeNode(&self.tree, expr)));
                self.match_token(TokenTag::Comma);
                expr
            }
        );
        let first = self.tree.node_index(range.start);
        let first_identifer = self.tree.node(first);
        if range.end - range.start == 1 {
            Ok(first)
        } else {
            self.add_node(
                Tag::Expressions,
                first_identifer.token,
                range.start,
                range.end,
            )
        }
    }

    // token: ':'
    // lhs: type_expr
    // rhs: init_expr
    fn parse_decl_variable(&mut self) -> Result<NodeId> {
        let identifier_list = self.parse_identifier_list()?;
        let token = self.shift_token(); // : or ::
        let mut type_expr = 0;
        let mut init_expr = 0;
        match self.tree.token(token).tag {
            TokenTag::Colon => {
                type_expr = self.parse_expr_base()?;
                if self.match_token(TokenTag::Equal) {
                    init_expr = self.parse_expr()?;
                }
            }
            TokenTag::ColonEqual => {
                init_expr = self.parse_expr_list()?;
            }
            _ => init_expr = self.parse_expr()?,
        }
        self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon])?;
        let variable_decl =
            self.add_node(Tag::VariableDecl, identifier_list, type_expr, init_expr)?;
        let identifier_node = self.tree.node_mut(identifier_list);
        // Link identifier nodes back to variable declaration.
        match identifier_node.tag {
            Tag::Identifier => identifier_node.lhs = variable_decl,
            Tag::Expressions => {
                for i in identifier_node.lhs..identifier_node.rhs {
                    let ni = self.tree.node_index(i);
                    let node = self.tree.node_mut(ni);
                    node.lhs = variable_decl;
                }
            }
            _ => unreachable!(),
        }
        Ok(variable_decl)
    }

    /// struct-decl = identifier :: struct field* end
    fn parse_decl_struct(&mut self) -> Result<NodeId> {
        self.expect_token(TokenTag::Identifier)?;
        self.expect_token(TokenTag::ColonColon)?;
        let token = self.expect_token(TokenTag::Struct)?;
        let type_parameters = if self.current_tag() == TokenTag::BraceL {
            self.parse_type_parameters()?
        } else {
            0
        };
        self.match_token(TokenTag::Newline);
        let mut source_index = 0;
        let range = parse_while!(self, self.token_isnt(TokenTag::End), {
            let field = self.parse_field(source_index)?;
            source_index += 1;
            self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon])?;
            field
        });
        self.expect_token_and_newline(TokenTag::End)?;
        let fields = self.add_indices_fixed(&[range.start, range.end]);
        self.add_node(Tag::Struct, token, type_parameters, fields)
    }

    // field = identifier ':' type ';'
    fn parse_field(&mut self, source_index: u32) -> Result<NodeId> {
        let identifier_token = self.expect_token(TokenTag::Identifier)?;
        let identifier = self.add_leaf(Tag::Identifier, identifier_token)?;
        let colon = self.expect_token(TokenTag::Colon)?;
        let type_expr = self.parse_expr_base()?;
        let extra_data = self.add_indices_fixed(&[type_expr, source_index]);
        self.add_node(Tag::Field, colon, identifier, extra_data)
    }

    /**************************************************************************/
    // Statements

    /// stmt    = block | break | continue | if | return | while
    ///         | var-decl | assign-stmt
    pub fn parse_stmt(&mut self) -> Result<NodeId> {
        let result = match self.current_tag() {
            TokenTag::Block => self.parse_stmt_block(),
            TokenTag::Break => {
                let token = self.shift_token();
                self.add_leaf(Tag::Break, token)
            }
            TokenTag::Continue => {
                let token = self.shift_token();
                self.add_leaf(Tag::Continue, token)
            }
            TokenTag::If => self.parse_stmt_if(),
            TokenTag::Return => {
                let token = self.shift_token();
                let range = parse_while!(
                    self,
                    self.token_isnt(TokenTag::Newline) && !self.token_is(TokenTag::Semicolon),
                    {
                        let expr = self.parse_expr()?;
                        self.match_token(TokenTag::Comma);
                        expr
                    }
                );
                let node = self.add_node(Tag::Return, token, range.start, range.end);
                self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon])?;
                node
            }
            TokenTag::While => self.parse_stmt_while(),
            _ => match self.next_tag(1) {
                TokenTag::Colon | TokenTag::ColonEqual | TokenTag::Comma => {
                    // self.shift_token();
                    self.parse_decl_variable()
                }
                _ => self.parse_stmt_assign(),
            },
        };
        match result {
            Ok(node_id) => Ok(node_id),
            Err(diagnostic) => {
                self.workspace.diagnostics.push(diagnostic);
                self.skip_to_next_statement();
                Ok(1)
            }
        }
    }

    // stmt-assign = expr assign-op expr
    fn parse_stmt_assign(&mut self) -> Result<NodeId> {
        let lhs = self.parse_expr()?;
        if lhs == 0 {
            return Ok(0);
        }
        let lhs_node = self.tree.node(lhs);
        if lhs_node.tag == Tag::IntegerLiteral {
            return Err(Diagnostic::error()
                .with_message("invalid left-hand side of assignment")
                .with_labels(vec![self.tree.label(lhs_node.token)]));
        }
        match self.current_tag() {
            TokenTag::Equal => {
                // expr '=' expr
                let op_token = self.shift_token();
                let rhs = self.parse_expr()?;
                self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon])?;
                self.add_node(Tag::Assign, op_token, lhs, rhs)
            }
            TokenTag::PlusEqual => {
                // expr '+=' expr
                let op_token = self.shift_token();
                let rhs = self.parse_expr()?;
                self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon])?;
                self.add_node(Tag::AssignAdd, op_token, lhs, rhs)
            }
            _ => {
                // expr
                self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon])?;
                Ok(lhs)
            }
        }
    }

    /// block = 'block' body
    fn parse_stmt_block(&mut self) -> Result<NodeId> {
        let block_token = self.shift_token();
        self.match_token(TokenTag::Newline);
        self.parse_stmt_body(block_token)
    }

    /// body = stmt* 'end'
    fn parse_stmt_body(&mut self, token: TokenId) -> Result<NodeId> {
        let range = parse_while!(self, self.token_isnt(TokenTag::End), {
            self.parse_stmt().unwrap()
        });
        self.expect_token_and_newline(TokenTag::End)?; // 'end'
        self.add_node(Tag::Block, token, range.start, range.end)
    }

    /// stmt-if
    fn parse_stmt_if(&mut self) -> Result<NodeId> {
        let if_token = self.current_id();

        let range = parse_while!(self, self.token_isnt(TokenTag::End), {
            // Parse condition
            let mut else_if_token = self.shift_token(); // 'else' | 'if'

            let condition = if else_if_token == if_token {
                // 'if'
                self.parse_expr()?
            } else if self.current_tag() == TokenTag::If {
                // 'else if'
                else_if_token = self.shift_token();
                self.parse_expr()?
            } else {
                // 'else'
                0
            };
            self.match_token(TokenTag::Newline);

            // Parse body
            let range = parse_while!(
                self,
                self.token_isnt(TokenTag::Else) && !self.token_is(TokenTag::End),
                { self.parse_stmt()? }
            );
            let block = self.add_node(Tag::Block, else_if_token, range.start, range.end)?;

            self.add_node(Tag::If, else_if_token, condition, block)?
        });

        self.expect_token_and_newline(TokenTag::End)?;

        if range.end - range.start == 1 {
            return Ok(self.tree.node_index(range.start));
        }
        self.add_node(Tag::IfElse, if_token, range.start, range.end)
    }

    /// stmt-while =
    fn parse_stmt_while(&mut self) -> Result<NodeId> {
        let token = self.shift_token();
        let condition = self.parse_expr()?;
        self.match_token(TokenTag::Newline);
        let range = parse_while!(self, self.token_isnt(TokenTag::End), { self.parse_stmt()? });
        self.expect_token_and_newline(TokenTag::End)?;
        let body = self.add_node(Tag::Block, 0, range.start, range.end)?;
        self.add_node(Tag::While, token, condition, body)
    }

    /**************************************************************************/

    // Expressions

    /// expr = expr-precedence
    pub fn parse_expr(&mut self) -> Result<NodeId> {
        self.parse_expr_precedence(0)
    }

    /// expr-precedence = expr-prefix (op expr-prefix)*
    fn parse_expr_precedence(&mut self, min_precedence: i32) -> Result<NodeId> {
        // Parse the left-hand side.
        let mut lhs = self.parse_expr_prefix()?;
        if lhs == 0 {
            return Ok(0);
        }
        let mut invalid = -1;
        loop {
            let token_tag = self.current_tag();
            let op = token_to_operator(token_tag);

            // Infix operators
            if op.precedence < min_precedence {
                break;
            }
            if op.precedence == invalid {
                panic!("chained comparison operator");
            }
            let op_token_index = self.shift_token();

            // Postfix operators
            // match op.tag {
            //     Tag::Factorial => {
            //         lhs = self.add_node(op.tag, op_token_index, lhs, 0);
            //         continue;
            //     }
            //     Tag::Subscript => {
            //         let rhs = self.parse_expr();
            //         self.expect_token(TokenTag::BracketR);
            //         lhs = self.add_node(Tag::Subscript, op_token_index, lhs, rhs);
            //         continue;
            //     }
            //     _ => {}
            // }
            // Recursively parse the right-hand side.
            let right_precedence = if let Associativity::Right = op.associativity {
                op.precedence - 1
            } else {
                op.precedence + 1
            };
            let rhs = self.parse_expr_precedence(right_precedence)?;
            if rhs == 0 {
                return Ok(0);
            }
            lhs = self.add_node(op.tag, op_token_index, lhs, rhs)?;

            if let Associativity::None = op.associativity {
                invalid = op.precedence;
            }
        }
        Ok(lhs)
    }

    /// expr-prefix = prefix-op* expr-operand
    fn parse_expr_prefix(&mut self) -> Result<NodeId> {
        let tag = match self.current_tag() {
            TokenTag::Ampersand => Tag::Address,
            TokenTag::Bang => Tag::Not,
            TokenTag::Minus => Tag::Negation,
            TokenTag::Tilde => Tag::BitwiseNot,
            // TokenTag::ParenL => {
            //     self.shift_token();
            //     let expr = self.parse_expr();
            //     self.expect_token(TokenTag::ParenR);
            //     return expr;
            // }
            _ => return self.parse_expr_postfix(),
        };
        let op_token = self.shift_token();
        let expr = self.parse_expr_prefix()?;
        self.add_node(tag, op_token, expr, 0)
    }

    /// expr-operand = expr-base ( '(' expressions ')' )?
    fn parse_expr_postfix(&mut self) -> Result<NodeId> {
        let mut lhs = self.parse_expr_base()?;
        loop {
            match self.current_tag() {
                TokenTag::At => {
                    let token = self.shift_token();
                    lhs = self.add_node(Tag::Dereference, token, lhs, 0)?;
                    continue;
                }
                TokenTag::Bang => {
                    let token = self.shift_token();
                    lhs = self.add_node(Tag::Factorial, token, lhs, 0)?;
                    continue;
                }
                TokenTag::BracketL => {
                    let token = self.shift_token();
                    let subscript = self.parse_expr()?;
                    self.expect_token(TokenTag::BracketR)?;
                    lhs = self.add_node(Tag::Subscript, token, lhs, subscript)?;
                    continue;
                }
                TokenTag::Dot => {
                    let token = self.expect_token(TokenTag::Dot)?;
                    let identifier_token = self.expect_token(TokenTag::Identifier)?;
                    let identifier = self.add_leaf(Tag::Identifier, identifier_token)?;
                    lhs = self.add_node(Tag::Access, token, lhs, identifier)?;
                    continue;
                }
                TokenTag::ParenL => {
                    let token = self.shift_token();
                    let range = parse_while!(self, self.token_isnt(TokenTag::ParenR), {
                        let expr = self.parse_expr()?;
                        self.match_token(TokenTag::Comma);
                        expr
                    });
                    self.expect_token(TokenTag::ParenR)?;
                    let expr_list = self.add_node(Tag::Expressions, 0, range.start, range.end)?;
                    return self.add_node(Tag::Call, token, lhs, expr_list);
                }
                _ => return Ok(lhs),
            }
        }
    }
    /// expr-base = IDENTIFIER | expr-group | LITERAL | type-base
    fn parse_expr_base(&mut self) -> Result<NodeId> {
        let token = self.shift_token();
        match self.tree.token(token).tag {
            TokenTag::Identifier => {
                // IDENTIFIER
                let identifier = self.add_leaf(Tag::Identifier, token);
                if self.match_token(TokenTag::BraceL) {
                    // IDENTIFIER '{' expr-list '}'
                    let range = parse_while!(self, self.token_isnt(TokenTag::BraceR), {
                        let expr = self.parse_expr()?;
                        self.match_token(TokenTag::Comma);
                        expr
                    });

                    self.expect_token(TokenTag::BraceR)?;
                    return self.add_node(Tag::Type, token, range.start, range.end);
                }
                identifier
            }
            TokenTag::ParenL => {
                //
                if self.next_tag(1) == TokenTag::ParenR && self.next_tag(2) == TokenTag::Arrow
                    || self.next_tag(1) == TokenTag::Identifier
                        && self.next_tag(2) == TokenTag::Colon
                {
                    return self.parse_prototype();
                }
                // '(' expr ')'
                let expr = self.parse_expr()?;
                self.expect_token(TokenTag::ParenR)?;
                Ok(expr)
            }
            TokenTag::IntegerLiteral => self.add_leaf(Tag::IntegerLiteral, token),
            TokenTag::True => self.add_leaf(Tag::True, token),
            TokenTag::False => self.add_leaf(Tag::False, token),
            TokenTag::StringLiteral => self.add_leaf(Tag::StringLiteral, token),
            _ => Err(Diagnostic::error()
                .with_message(format!(
                    "expected expression, got {:?}.",
                    self.tree.token_str(token)
                ))
                .with_labels(vec![self.tree.label(token)])),
        }
    }

    // Nodes

    fn add_node(&mut self, tag: Tag, token: TokenId, lhs: u32, rhs: u32) -> Result<NodeId> {
        self.tree.nodes.push(Node {
            tag,
            token,
            lhs,
            rhs,
        });
        Ok((self.tree.nodes.len() - 1) as NodeId)
    }

    fn add_leaf(&mut self, tag: Tag, token: TokenId) -> Result<NodeId> {
        self.add_node(tag, token, 0, 0)
    }

    // fn parse_while<C, F>(&mut self, cond: C, parse_fn: F) -> Result<Range<u32>>
    // where
    //     C: Fn(&Self) -> bool,
    //     F: Fn(&mut Self) -> Result<NodeId>,
    // {
    //     let stack_top = self.stack.len();
    //     while cond(self) {
    //         let node = parse_fn(self)?;
    //         self.stack.push(node);
    //     }
    //     Ok(self.add_indices(stack_top))
    // }

    fn add_indices(&mut self, stack_top: usize) -> Range<u32> {
        let start = self.tree.indices.len() as u32;
        for i in stack_top..self.stack.len() {
            self.tree.indices.push(self.stack[i]);
        }
        let end = self.tree.indices.len() as u32;
        self.stack.truncate(stack_top);
        Range { start, end }
    }

    fn add_indices_fixed(&mut self, indices: &[u32]) -> u32 {
        let start = self.tree.indices.len() as u32;
        for &i in indices {
            self.tree.indices.push(i);
        }
        start
    }

    /**************************************************************************/
    // Tokens

    fn current_id(&self) -> TokenId {
        self.index as TokenId
    }

    fn current_token(&self) -> &Token {
        self.tree.token(self.index as TokenId)
    }

    /// Gets the current token tag.
    fn current_tag(&self) -> TokenTag {
        self.tree.tokens[self.index].tag
    }

    fn previous_tag(&self) -> Option<TokenTag> {
        if self.index == 0 {
            None
        } else {
            Some(self.tree.tokens[self.index - 1].tag)
        }
    }

    /// Gets the token tag at the given offset ahead.
    fn next_tag(&self, offset: usize) -> TokenTag {
        assert!(
            self.index + offset < self.tree.tokens.len(),
            "unexpected end-of-file while parsing"
        );
        self.tree.tokens[self.index + offset].tag
    }

    /// Consumes and returns the token at the current index.
    fn shift_token(&mut self) -> TokenId {
        let result = self.index;
        self.index += 1;
        result as TokenId
    }

    /// Consumes the current token if its tag matches.
    fn match_token(&mut self, tag: TokenTag) -> bool {
        if self.token_is(tag) {
            self.index += 1;
            return true;
        }
        false
    }

    // fn skip_to_next_root_declaration(&mut self) {
    //     while !self.is_at_end() {
    //         if self.current_tag() == TokenTag::Newline
    //             && self.next_tag(1) == TokenTag::Identifier
    //             && self.next_tag(2) == TokenTag::ColonColon
    //         {
    //             let token_id = self.current_id() + 1;
    //             let token = self.tree.tokens[token_id as usize];
    //             let source = self.tree.token_source(token_id).0;
    //             let byte_before_identifier = source.as_bytes()[token.start as usize - 1];
    //             if byte_before_identifier == b'\n' {
    //                 self.shift_token();
    //                 return;
    //             }
    //         }
    //         self.shift_token();
    //     }
    // }

    fn skip_to_next_statement(&mut self) {
        while !self.is_at_end() {
            match self.current_tag() {
                TokenTag::Block
                | TokenTag::Break
                | TokenTag::Continue
                | TokenTag::If
                | TokenTag::Return
                | TokenTag::While => return,
                TokenTag::Identifier
                    if self.previous_tag().expect("cannot get previous token tag")
                        == TokenTag::Newline =>
                {
                    return;
                }
                _ => match self.next_tag(1) {
                    TokenTag::Colon | TokenTag::ColonEqual | TokenTag::Comma | TokenTag::Equal => {
                        dbg!(self.current_tag());
                        return;
                    }
                    _ => self.shift_token(),
                },
            };
        }
    }

    fn is_at_end(&self) -> bool {
        self.index >= self.tree.tokens.len() - 1 || self.token_is(TokenTag::Eof)
    }

    fn assert_token(&mut self, tag: TokenTag) -> Result<()> {
        if self.token_is(tag) {
            return Ok(());
        }
        let token_id = self.current_id();
        let token = self.tree.token(token_id);
        Err(Diagnostic::error().with_labels(vec![self
            .tree
            .label(token_id)
            .with_message(format!("expected token {:?}, got {:?}.", tag, token.tag))]))
    }

    // fn assert_token_order(&mut self, tags: &[TokenTag]) {
    //     for i in 0..tags.len() {
    //         assert!(
    //             self.next_tag(i) == tags[i],
    //             "Error: expected tokens {:?}, got {:?}. Token index: {:?}.",
    //             tags,
    //             self.next_tag(i),
    //             self.index + i
    //         );
    //     }
    // }

    fn assert_tokens(&mut self, tags: &[TokenTag], offset: usize) -> Result<()> {
        for tag in tags {
            if self.next_tag(offset) == *tag {
                return Ok(());
            }
        }
        let token_id = self.current_id();
        let token = self.current_token();
        Err(Diagnostic::error().with_labels(vec![self
            .tree
            .label(token_id)
            .with_message(format!("expected one of {:?}, got {:?}.", tags, token.tag))]))
    }

    fn expect_token(&mut self, tag: TokenTag) -> Result<TokenId> {
        self.assert_token(tag)?;
        Ok(self.shift_token())
    }

    fn expect_tokens(&mut self, tags: &[TokenTag]) -> Result<TokenId> {
        self.assert_tokens(tags, 0)?;
        Ok(self.shift_token())
    }

    fn expect_token_and_newline(&mut self, tag: TokenTag) -> Result<TokenId> {
        let token = self.expect_token(tag)?;
        self.match_token(TokenTag::Newline);
        Ok(token)
    }

    fn token_is(&self, tag: TokenTag) -> bool {
        self.tree.tokens[self.index].tag == tag
    }

    fn token_isnt(&self, tag: TokenTag) -> bool {
        !self.token_is(tag)
    }
}

const SPACES: usize = 2;

fn write_indent(f: &mut fmt::Formatter, indentation: usize) -> fmt::Result {
    write!(f, "{1:0$}", indentation * SPACES, "")
}

pub struct Tree {
    pub sources: Vec<String>,
    pub tokens: Vec<Token>,
    pub nodes: Vec<Node>,
    pub indices: Vec<u32>,
    pub modules: Vec<Module>,
}

impl Tree {
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id as usize]
    }

    pub fn lchild(&self, parent: &Node) -> &Node {
        &self.nodes[parent.lhs as usize]
    }

    pub fn rchild(&self, parent: &Node) -> &Node {
        &self.nodes[parent.rhs as usize]
    }

    pub fn token(&self, id: TokenId) -> &Token {
        &self.tokens[id as usize]
    }

    pub fn node_index(&self, index: u32) -> NodeId {
        self.indices[index as usize] as NodeId
    }

    pub fn node_indirect(&self, index: u32) -> &Node {
        &self.nodes[self.node_index(index) as usize]
    }

    pub fn node_token(&self, node: &Node) -> &Token {
        &self.tokens[node.token as usize]
    }

    pub fn node_full_name(&self, id: NodeId) -> String {
        let node = self.node(id);
        let node_name = self.name(id);
        let module_token_id = self.token_source(node.token).1;
        if let Some(module) = self.token_module(module_token_id) {
            if module.name == "GLFW" {
                return node_name.to_string();
            }
            format!("{}.{}", &module.name, node_name)
        } else {
            node_name.to_string()
        }
    }

    pub fn node_token_line(&self, id: NodeId) -> i64 {
        let node = self.node(id);
        let token = self.node_token(node);
        let source = self.token_source(node.token).0;
        token.line(source)
    }

    pub fn node_lexeme(&self, id: NodeId) -> &str {
        let node = self.node(id);
        let token = self.node_token(node);
        let source = self.token_source(node.token).0;
        token.to_str(source)
    }

    pub fn range(&self, node: &Node) -> Range<u32> {
        match node.tag {
            Tag::Struct => self.node_index(node.rhs)..self.node_index(node.rhs + 1),
            _ => node.lhs..node.rhs,
        }
    }

    pub fn token_str(&self, token_id: TokenId) -> &str {
        let token = self.tokens[token_id as usize];
        let source = self.token_source(token_id).0;
        token.to_str(source)
    }

    pub fn name(&self, id: NodeId) -> &str {
        let node = self.node(id);
        let token_index = match node.tag {
            Tag::Access => node.token as i32 + 1,
            Tag::Field => node.token as i32 - 1,
            Tag::FunctionDecl => node.token as i32 - 1,
            Tag::Module => node.token as i32 + 1,
            Tag::Struct => node.token as i32 - 2,
            _ => node.token as i32,
        } as usize;
        let token = &self.tokens[token_index];
        let source = self.token_source(node.token).0;
        token.to_str(source)
    }

    pub fn get_module_index(&self, module_name: &str) -> Option<usize> {
        for (i, entry) in self.modules.iter().enumerate() {
            if entry.name == module_name {
                return Some(i);
            }
        }
        None
    }

    pub fn node_lexeme_offset(&self, node: &Node, offset: i32) -> &str {
        if (node.token as i32 + offset) < 0 {
            panic!("bad offset")
        }
        let token = &self.tokens[(node.token as i32 + offset) as usize];
        let source = self.token_source(node.token).0;
        token.to_str(source)
    }

    pub fn token_module(&self, token_id: TokenId) -> Option<&Module> {
        if token_id == 0 {
            return None;
        }
        let module_index = self
            .modules
            .partition_point(|m| m.first_token_id < token_id);
        Some(&self.modules[module_index])
    }

    fn token_source(&self, token_id: TokenId) -> (&str, TokenId) {
        let mut source_index = 0;
        let mut last_module_token_index = 0;
        for m in self.modules.iter().skip(1) {
            if (token_id) < m.first_token_id {
                break;
            }
            source_index += 1;
            last_module_token_index = m.first_token_id;
        }
        (
            &self.sources[source_index],
            last_module_token_index as TokenId,
        )
    }

    pub fn source_id(&self, token_id: TokenId) -> usize {
        let mut source_id = 0;
        for m in self.modules.iter().skip(1) {
            if (token_id) < m.first_token_id {
                break;
            }
            source_id += 1;
        }
        source_id
    }

    pub fn label(&self, token_id: TokenId) -> Label<usize> {
        Label::primary(self.source_id(token_id), self.token(token_id).range())
    }

    /**************************************************************************/
    // Debug

    fn print_node(&self, f: &mut fmt::Formatter, id: NodeId, indentation: usize) -> fmt::Result {
        let node = self.node(id);
        let tag = node.tag;
        if node.lhs == 0 && node.rhs == 0 {
            if tag == Tag::Parameters {
                write!(f, "()")?;
            } else {
                write!(f, "{}", self.node_lexeme(id))?;
            }
            return Ok(());
        }

        match tag {
            // Multiple children, multiple lines.
            Tag::Block
            | Tag::Expressions
            | Tag::IfElse
            | Tag::Module
            | Tag::Parameters
            | Tag::Root
            | Tag::Struct => {
                // if tag == Tag::Root {
                //     for i in node.lhs..node.rhs {
                //         let module_node = self.node(self.node_index(i));
                //         write!(f, "Module {}: ", self.node_index(i));
                //         for j in module_node.lhs..module_node.rhs {
                //             write!(f, "{}, ", self.node_index(j));
                //         }
                //         writeln!(f);
                //     }
                // }
                writeln!(f, "({:?}", tag)?;
                for i in self.range(node) {
                    write_indent(f, indentation + 1)?;
                    self.print_node(f, self.node_index(i), indentation + 1)?;
                    writeln!(f)?;
                }
                write_indent(f, indentation)?;
                write!(f, ")")?;
            }
            // Multiple children, single line.
            Tag::Return | Tag::Type | Tag::TypeParameters => {
                write!(f, "({:?}", tag)?;
                for i in node.lhs..node.rhs {
                    write!(f, " ")?;
                    self.print_node(f, self.node_index(i), indentation + 1)?;
                }
                write!(f, ")")?;
            }
            // One or two children.
            _ => {
                write!(f, "({:?}", tag)?;
                if tag == Tag::Identifier {
                    return Ok(());
                }
                if node.lhs != 0 {
                    write!(f, " ")?;
                    self.print_node(f, node.lhs, indentation)?;
                }
                if tag == Tag::Field {
                    write!(f, " ")?;
                    self.print_node(f, self.node_index(node.rhs), indentation)?;
                } else if node.rhs != 0 && tag != Tag::Grouping {
                    write!(f, " ")?;
                    self.print_node(f, node.rhs, indentation)?;
                }
                write!(f, ")")?;
            }
        }
        Ok(())
    }
}

struct TreeNode<'a>(&'a Tree, NodeId);

impl<'a> fmt::Display for TreeNode<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.print_node(f, self.1, 0)?;
        Ok(())
    }
}

impl fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let root = self.node(0);
        if root.lhs == 0 && root.rhs == 0 {
            self.print_node(f, (self.nodes.len() - 1) as u32, 0)?;
        } else {
            self.print_node(f, 0_u32, 0)?;
        };
        Ok(())
    }
}
