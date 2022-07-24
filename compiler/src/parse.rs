use crate::tokenize::{Tag as TokenTag, Token, Tokenizer};
use crate::utils::assert_size;
use std::fmt;
use std::fmt::Debug;

use std::ops::Range;

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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Tag {
    Access,         // lhs, rhs
    Add,            // lhs, rhs
    Address,        // expr
    LogicalAnd,     // lhs, rhs
    LogicalOr,      // lhs, rhs
    Assign,         // lhs, rhs
    AssignAdd,      // lhs, rhs
    Block,          // start..end [Stmt]
    BlockDirect,    //
    BitwiseAnd,     // expr
    BitwiseNot,     // expr
    BitwiseOr,      // lhs, rhs
    BitwiseXor,     // lhs, rhs
    Break,          //
    Call,           // func_expr, arguments: Expressions
    Continue,       //
    Equality,       // lhs, rhs
    Expressions,    // start..end [Expr]
    Dereference,    // expr
    Factorial,      // expr
    Field,          // type_expr
    FunctionDecl,   // prototype, block
    Greater,        // lhs, rhs
    Grouping,       // expr
    Identifier,     //
    If,             // condition, block
    IfElse,         // start..end [If]
    Import,         // lhs, rhs
    Inequality,     // lhs, rhs
    IntegerLiteral, //
    Invalid,        //
    Less,           // lhs, rhs
    Div,            // lhs, rhs
    Module,         // start..end [Declaration]
    Mul,            // lhs, rhs
    Negation,       // expr
    Not,            // expr
    Parameters,     // start..end [Field]
    Prototype,      // parameters: Parameters, returns: Expressions
    Return,         // expr
    Root,           // start..end [Decl]
    StringLiteral,  //
    Struct,         // start..end [Field]
    Sub,            // lhs, rhs
    Subscript,      // lhs, rhs
    Type,           // expr
    VariableDecl,   // type_expr, init_expr
    While,          // condition, block
}

pub struct Node2 {
    token: u32,
    data: Data,
}

pub enum Data {
    Add { lhs: u32, rhs: u32 },
    And { lhs: u32, rhs: u32 },
    Assign { lhs: u32, rhs: u32 },
    Block { start: u32, end: u32 },
    BlockDirect,
    BitwiseAnd { lhs: u32, rhs: u32 },
    BitwiseNot { expr: u32 },
    BitwiseOr { lhs: u32, rhs: u32 },
    BitwiseXor { lhs: u32, rhs: u32 },
    Break,
    Call { function: u32, arguments: u32 },
    Continue,
    Div { lhs: u32, rhs: u32 },
    Expressions { start: u32, end: u32 },
    Field { type_expr: u32 },
    FunctionDecl { prototype: u32, block: u32 },
    Greater { lhs: u32, rhs: u32 },
    Grouping { expr: u32 },
    Identifier,
    If { condition: u32, block: u32 },
    IfElse { start: u32, end: u32 },
    IntegerLiteral,
    Invalid,
    Less { lhs: u32, rhs: u32 },
    Mul { lhs: u32, rhs: u32 },
    Negation { expr: u32 },
    Not { expr: u32 },
    Parameters { start: u32, end: u32 },
    Prototype { parameters: u32, returns: u32 },
    Return { expr: u32 },
    Root { start: u32, end: u32 },
    StringLiteral,
    Struct { start: u32, end: u32 },
    Sub { lhs: u32, rhs: u32 },
    TypeBase { expr: u32 },
    VariableDecl { type_expr: u32, init_expr: u32 },
    While { condition: u32, block: u32 },
}

// Assert that Node size <= 16 bytes
pub const _ASSERT_NODE_SIZE: () = assert_size::<Node>(16);
pub const _ASSERT_NODE_2_SIZE: () = assert_size::<Node2>(16);
pub const _ASSERT_NODE_DATA_SIZE: () = assert_size::<Data>(12);

type TokenId = u32;
pub type NodeId = u32;
pub type Id = u32;

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

pub struct Parser {
    /// Current token index.
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

impl Parser {
    pub fn new(source: &str, tokens: Vec<Token>) -> Self {
        Self {
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
                module_indices: Vec::new(),
            },
        }
    }

    pub fn parse(&mut self) -> u32 {
        self.parse_modules();
        0
    }

    pub fn tree(self) -> Tree {
        self.tree
    }

    /**************************************************************************/
    // Declarations

    /// program = module*
    fn parse_modules(&mut self) -> Range<NodeId> {
        let stack_top = self.stack.len();
        while self.index < self.tree.tokens.len() - 1 {
            let module = self.parse_module();
            self.stack.push(module);
        }
        let range = self.add_indices(stack_top);
        self.tree.nodes[0] = Node {
            tag: Tag::Root,
            token: 0,
            lhs: range.start,
            rhs: range.end,
        };
        range
    }

    /// module = root-decl*
    fn parse_module(&mut self) -> NodeId {
        self.match_token(TokenTag::Newline);
        let range = self.parse_until(TokenTag::Eof, Self::parse_root_declaration);
        self.expect_token(TokenTag::Eof);
        self.add_node(Tag::Module, 0, range.start, range.end)
    }

    /// root-decl = fn-decl | var-decl | struct-decl | module-decl
    fn parse_root_declaration(&mut self) -> NodeId {
        self.match_token(TokenTag::Newline);
        self.assert_token(TokenTag::Identifier);
        // self.assert_tokens(&[TokenTag::Colon, TokenTag::ColonColon], 1);
        // Look at the token after the '::' or ':'.
        match self.next_token_tag(2) {
            TokenTag::ParenL => {
                // identifier :: ()
                if self.next_token_tag(3) == TokenTag::ParenR {
                    return self.parse_decl_function();
                }
                // identifier :: (parameter: ...
                if self.next_token_tag(3) == TokenTag::Identifier
                    && self.next_token_tag(4) == TokenTag::Colon
                {
                    return self.parse_decl_function();
                }
                // identifier :: ( ...
                return self.parse_decl_variable();
            }
            // TokenTag::Function => return self.parse_decl_function(),
            // identifier :: struct ...
            TokenTag::Module => return self.parse_module_import(),
            TokenTag::Struct => return self.parse_decl_struct(),
            // identifier :: ...
            _ => return self.parse_decl_variable(),
        }
    }

    /// module-import = identifier '::' 'module' string-literal
    fn parse_module_import(&mut self) -> NodeId {
        let alias_token = self.expect_token(TokenTag::Identifier);
        self.expect_token(TokenTag::ColonColon);
        let module_token = self.expect_token(TokenTag::Module);
        let name_token = self.expect_token(TokenTag::StringLiteral);
        let lhs = self.add_leaf(Tag::Identifier, alias_token);
        let rhs = self.add_leaf(Tag::StringLiteral, name_token);

        let current_source = self.tree.token_source(name_token);
        let module_name = self
            .tree
            .token(name_token)
            .to_str(current_source)
            .trim_matches('"')
            .to_string();

        if !self.is_module_tokenized(&module_name) {
            let filename = &format!("{}.hb", module_name);
            let path = std::path::Path::new("modules").join(filename);
            if path.exists() {
                self.tree
                    .module_indices
                    .push((module_name, self.tree.tokens.len()));
                let source = std::fs::read_to_string(path).unwrap();
                let mut tokenizer = Tokenizer::new(&source);
                tokenizer.append_tokens(&mut self.tree.tokens);
                self.tree.sources.push(source);
                println!("{:?}", self.tree.module_indices);
            } else {
                println!("Path doesn't exist.");
            }
        }

        self.match_token(TokenTag::Newline);
        self.add_node(Tag::Import, module_token, lhs, rhs)
    }

    fn is_module_tokenized(&self, module_name: &str) -> bool {
        for entry in &self.tree.module_indices {
            if entry.0 == module_name {
                return true;
            }
        }
        false
    }

    /// function-decl = ideentifier '::' '(' ')'
    fn parse_decl_function(&mut self) -> NodeId {
        self.expect_token(TokenTag::Identifier); // identifier
        let token_index = self.expect_token(TokenTag::ColonColon); // '::'
        let prototype = self.parse_prototype();
        self.expect_token(TokenTag::Newline);
        let body = self.parse_stmt_body(0);
        self.add_node(Tag::FunctionDecl, token_index, prototype, body)
    }

    /// prototype =
    fn parse_prototype(&mut self) -> NodeId {
        let parameters = self.parse_parameters();
        let returns = if self.match_token(TokenTag::Arrow) {
            self.parse_type_list()
        } else {
            0
        };
        self.add_node(Tag::Prototype, 0, parameters, returns)
    }

    /// parameters =
    fn parse_parameters(&mut self) -> NodeId {
        let token = self.expect_token(TokenTag::ParenL);
        let range = self.parse_until(TokenTag::ParenR, Self::parse_parameter);
        self.expect_token(TokenTag::ParenR);
        self.add_node(Tag::Parameters, token, range.start, range.end)
    }

    // parameter
    fn parse_parameter(&mut self) -> NodeId {
        let identifier = self.expect_token(TokenTag::Identifier);
        self.expect_token(TokenTag::Colon);
        let type_expr = self.parse_expr_base();
        self.match_token(TokenTag::Comma);
        self.add_node(Tag::Field, identifier, type_expr, 0)
    }

    /// type-list = expr-base | '(' (expr-base ',')* ')'
    fn parse_type_list(&mut self) -> NodeId {
        if self.match_token(TokenTag::ParenL) {
            let range = self.parse_until(TokenTag::ParenR, |s: &mut Self| -> NodeId {
                let type_expr = s.parse_expr_base();
                s.match_token(TokenTag::Comma);
                type_expr
            });
            self.expect_token(TokenTag::ParenR);
            self.add_node(Tag::Expressions, 0, range.start, range.end)
        } else {
            self.parse_expr_base()
        }
    }

    // token: ':'
    // lhs: type_expr
    // rhs: init_expr
    fn parse_decl_variable(&mut self) -> NodeId {
        self.expect_token(TokenTag::Identifier);
        let token = self.shift_token(); // : or ::
        let mut type_expr = 0;
        let mut init_expr = 0;
        match self.tree.token(token).tag {
            TokenTag::Colon => {
                type_expr = self.parse_expr_base();
                if self.match_token(TokenTag::Equal) {
                    init_expr = self.parse_expr();
                }
            }
            TokenTag::ColonEqual => {
                init_expr = self.parse_expr();
            }
            _ => init_expr = self.parse_expr(),
        }
        self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon]);
        self.add_node(Tag::VariableDecl, token, type_expr, init_expr)
    }

    /// struct-decl = identifier :: struct field* end
    fn parse_decl_struct(&mut self) -> NodeId {
        self.expect_token(TokenTag::Identifier);
        self.expect_token(TokenTag::ColonColon);
        let token = self.expect_token(TokenTag::Struct);
        self.match_token(TokenTag::Newline);
        let range = self.parse_until(TokenTag::End, Self::parse_field);
        self.expect_token_and_newline(TokenTag::End);
        self.add_node(Tag::Struct, token, range.start, range.end)
    }

    // field = identifier ':' type ';'
    fn parse_field(&mut self) -> NodeId {
        let identifier = self.expect_token(TokenTag::Identifier);
        self.expect_token(TokenTag::Colon);
        let type_expr = self.parse_expr_base();
        self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon]);
        self.add_node(Tag::Field, identifier, type_expr, 0)
    }

    /**************************************************************************/
    // Statements

    /// stmt    = block | break | continue | if | return | while
    ///         | var-decl | assign-stmt
    pub fn parse_stmt(&mut self) -> NodeId {
        match self.current_token_tag() {
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
                let expr = self.parse_expr();
                let node = self.add_node(Tag::Return, token, expr, 0);
                self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon]);
                node
            }
            TokenTag::While => self.parse_stmt_while(),
            _ => match self.next_token_tag(1) {
                TokenTag::Colon | TokenTag::ColonEqual => {
                    // self.shift_token();
                    self.parse_decl_variable()
                }
                _ => self.parse_stmt_assign(),
            },
        }
    }

    // stmt-assign =
    fn parse_stmt_assign(&mut self) -> NodeId {
        let lhs = self.parse_expr();
        if lhs == 0 {
            return 0;
        }
        match self.current_token_tag() {
            TokenTag::Equal => {
                // expr '=' expr
                let op_token = self.shift_token();
                let rhs = self.parse_expr();
                self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon]);
                self.add_node(Tag::Assign, op_token, lhs, rhs)
            }
            TokenTag::PlusEqual => {
                // expr '+=' expr
                let op_token = self.shift_token();
                let rhs = self.parse_expr();
                self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon]);
                self.add_node(Tag::AssignAdd, op_token, lhs, rhs)
            }
            _ => {
                // expr
                self.expect_tokens(&[TokenTag::Newline, TokenTag::Semicolon]);
                lhs
            }
        }
    }

    /// block = 'block' body
    fn parse_stmt_block(&mut self) -> NodeId {
        let block_token = self.shift_token();
        self.match_token(TokenTag::Newline);
        self.parse_stmt_body(block_token)
    }

    /// body = stmt* 'end'
    fn parse_stmt_body(&mut self, token: TokenId) -> NodeId {
        let stack_top = self.stack.len();
        while self.token_isnt(TokenTag::End) {
            let stmt = self.parse_stmt();
            self.stack.push(stmt);
        }
        self.expect_token_and_newline(TokenTag::End); // 'end'
        if self.stack.len() == stack_top {
            self.add_leaf(Tag::BlockDirect, token)
        } else {
            let range = self.add_indices(stack_top);
            self.add_node(Tag::Block, token, range.start, range.end)
        }
    }

    /// stmt-if
    fn parse_stmt_if(&mut self) -> NodeId {
        let if_token = self.current_token();
        let outer_stack_top = self.stack.len();

        while self.token_isnt(TokenTag::End) {
            // Parse condition
            let mut else_if_token = self.shift_token(); // 'else' | 'if'

            let condition = if else_if_token == if_token {
                // 'if'
                self.parse_expr()
            } else if self.current_token_tag() == TokenTag::If {
                // 'else if'
                else_if_token = self.shift_token();
                self.parse_expr()
            } else {
                // 'else'
                0
            };
            self.match_token(TokenTag::Newline);

            // Parse body
            let stack_top = self.stack.len();
            while self.token_isnt(TokenTag::Else) && self.token_isnt(TokenTag::End) {
                let stmt = self.parse_stmt();
                self.stack.push(stmt);
            }
            let range = self.add_indices(stack_top);
            let block = self.add_node(Tag::Block, else_if_token, range.start, range.end);

            let if_stmt = self.add_node(Tag::If, else_if_token, condition, block);
            self.stack.push(if_stmt);
        }

        self.expect_token_and_newline(TokenTag::End);

        let range = self.add_indices(outer_stack_top);
        if range.end - range.start == 1 {
            return self.tree.node_index(range.start);
        }
        self.add_node(Tag::IfElse, if_token, range.start, range.end)
    }

    /// stmt-while =
    fn parse_stmt_while(&mut self) -> NodeId {
        let token = self.shift_token();
        let condition = self.parse_expr();
        self.match_token(TokenTag::Newline);
        let stack_top = self.stack.len();
        while self.token_isnt(TokenTag::End) {
            let stmt = self.parse_stmt();
            self.stack.push(stmt);
        }
        self.expect_token_and_newline(TokenTag::End);
        let range = self.add_indices(stack_top);
        let body = self.add_node(Tag::Block, 0, range.start, range.end);
        self.add_node(Tag::While, token, condition, body)
    }

    /**************************************************************************/

    // Expressions

    /// expr = expr-precedence
    pub fn parse_expr(&mut self) -> NodeId {
        self.parse_expr_precedence(0)
    }

    /// expr-precedence = expr-prefix (op expr-prefix)*
    fn parse_expr_precedence(&mut self, min_precedence: i32) -> NodeId {
        // Parse the left-hand side.
        let mut lhs = self.parse_expr_prefix();
        if lhs == 0 {
            return 0;
        }
        let mut invalid = -1;
        loop {
            let token_tag = self.current_token_tag();
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
            let rhs = self.parse_expr_precedence(right_precedence);
            if rhs == 0 {
                return 0;
            }
            lhs = self.add_node(op.tag, op_token_index, lhs, rhs);

            if let Associativity::None = op.associativity {
                invalid = op.precedence;
            }
        }
        lhs
    }

    /// expr-prefix = prefix-op* expr-operand
    fn parse_expr_prefix(&mut self) -> NodeId {
        let tag;
        match self.current_token_tag() {
            TokenTag::Ampersand => tag = Tag::Address,
            TokenTag::At => tag = Tag::Dereference,
            TokenTag::Bang => tag = Tag::Not,
            TokenTag::Minus => tag = Tag::Negation,
            TokenTag::Tilde => tag = Tag::BitwiseNot,
            // TokenTag::ParenL => {
            //     self.shift_token();
            //     let expr = self.parse_expr();
            //     self.expect_token(TokenTag::ParenR);
            //     return expr;
            // }
            _ => return self.parse_expr_postfix(),
        }
        let op_token = self.shift_token();
        let expr = self.parse_expr_prefix();
        self.add_node(tag, op_token, expr, 0)
    }

    /// expr-operand = expr-base ( '(' expressions ')' )?
    fn parse_expr_postfix(&mut self) -> NodeId {
        let mut lhs = self.parse_expr_base();
        loop {
            match self.current_token_tag() {
                TokenTag::Bang => {
                    let token = self.shift_token();
                    lhs = self.add_node(Tag::Factorial, token, lhs, 0);
                    continue;
                }
                TokenTag::BracketL => {
                    let token = self.shift_token();
                    let subscript = self.parse_expr();
                    self.expect_token(TokenTag::BracketR);
                    lhs = self.add_node(Tag::Subscript, token, lhs, subscript);
                    continue;
                }
                TokenTag::Dot => {
                    let token = self.expect_token(TokenTag::Dot);
                    let identifier_token = self.expect_token(TokenTag::Identifier);
                    let identifier = self.add_leaf(Tag::Identifier, identifier_token);
                    lhs = self.add_node(Tag::Access, token, lhs, identifier);
                    continue;
                }
                TokenTag::ParenL => {
                    let token = self.shift_token();
                    let range = self.parse_until(TokenTag::ParenR, |s: &mut Self| -> u32 {
                        let expr = s.parse_expr();
                        s.match_token(TokenTag::Comma);
                        expr
                    });
                    self.expect_token(TokenTag::ParenR);
                    let expr_list = self.add_node(Tag::Expressions, 0, range.start, range.end);
                    return self.add_node(Tag::Call, token, lhs, expr_list);
                }
                _ => return lhs,
            }
        }
    }
    /// expr-base = IDENTIFIER | expr-group | LITERAL | type-base
    fn parse_expr_base(&mut self) -> NodeId {
        let token = self.shift_token();
        match self.tree.token(token).tag {
            TokenTag::Identifier => {
                // IDENTIFIER
                let identifier = self.add_leaf(Tag::Identifier, token);
                if self.match_token(TokenTag::BraceL) {
                    // IDENTIFIER '{' expr-list '}'
                    let stack_top = self.stack.len();
                    while self.token_isnt(TokenTag::BraceR) {
                        let expr = self.parse_expr();
                        self.match_token(TokenTag::Comma);
                        self.stack.push(expr);
                    }
                    let range = self.add_indices(stack_top);

                    self.expect_token(TokenTag::BraceR);
                    return self.add_node(Tag::Type, token, range.start, range.end);
                }
                identifier
            }
            TokenTag::ParenL => {
                //
                if self.next_token_tag(1) == TokenTag::ParenR
                    && self.next_token_tag(2) == TokenTag::Arrow
                    || self.next_token_tag(1) == TokenTag::Identifier
                        && self.next_token_tag(2) == TokenTag::Colon
                {
                    return self.parse_prototype();
                }
                // '(' expr ')'
                let expr = self.parse_expr();
                self.expect_token(TokenTag::ParenR);
                expr
            }
            TokenTag::IntegerLiteral => self.add_leaf(Tag::IntegerLiteral, token),
            TokenTag::StringLiteral => self.add_leaf(Tag::StringLiteral, token),
            _ => self.add_leaf(Tag::Identifier, token),
        }
    }

    // Nodes

    fn add_node(&mut self, tag: Tag, token: TokenId, lhs: u32, rhs: u32) -> NodeId {
        self.tree.nodes.push(Node {
            tag,
            token,
            lhs,
            rhs,
        });
        (self.tree.nodes.len() - 1) as NodeId
    }

    fn add_leaf(&mut self, tag: Tag, token: TokenId) -> NodeId {
        self.add_node(tag, token, 0, 0)
    }

    /// Returns a half-open range of node indices.
    fn parse_until<F>(&mut self, tag: TokenTag, parse_fn: F) -> Range<NodeId>
    where
        F: Fn(&mut Self) -> NodeId,
    {
        let stack_top = self.stack.len();
        while self.token_isnt(tag) {
            let node = parse_fn(self);
            self.stack.push(node);
        }
        self.add_indices(stack_top)
    }

    // fn parse_span_while<C, F>(&mut self, cond: C, tag: TokenTag, parse_fn: F) -> Range<u32>
    // where
    //     C: Fn(&mut Self) -> bool,
    //     F: Fn(&mut Self) -> u32,
    // {
    //     let stack_top = self.stack.len();
    //     while cond(self) {
    //         let node = parse_fn(self);
    //         self.stack.push(node);
    //     }
    //     self.add_indices(stack_top)
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

    // fn add_indices2(&mut self, a: u32, b: u32) -> u32 {
    //     let start = self.indices.len() as u32;
    //     self.indices.push(a);
    //     self.indices.push(b);
    //     start
    // }

    // fn add_indices3(&mut self, a: u32, b: u32, c: u32) -> u32 {
    //     let start = self.indices.len() as u32;
    //     self.indices.push(a);
    //     self.indices.push(b);
    //     self.indices.push(c);
    //     start
    // }

    /**************************************************************************/
    // Tokens

    /// Gets the current token tag.
    fn current_token_tag(&self) -> TokenTag {
        self.tree.tokens[self.index].tag
    }

    /// Gets the token tag at the given offset ahead.
    fn next_token_tag(&self, offset: usize) -> TokenTag {
        assert!(
            self.index + offset < self.tree.tokens.len(),
            "unexpected end-of-file while parsing"
        );
        self.tree.tokens[self.index + offset].tag
    }

    fn current_token(&self) -> TokenId {
        self.index as TokenId
    }

    // fn previous_token(&self) -> TokenId {
    //     (self.index - 1) as TokenId
    // }

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

    fn assert_token(&mut self, tag: TokenTag) {
        assert_eq!(
            self.token_is(tag),
            true,
            "Error: expected token {:?}, got {:?}. Token index: {:?}.",
            tag,
            self.current_token_tag(),
            self.index
        );
    }

    fn assert_token_order(&mut self, tags: &[TokenTag]) {
        for i in 0..tags.len() {
            assert!(
                self.next_token_tag(i) == tags[i],
                "Error: expected tokens {:?}, got {:?}. Token index: {:?}.",
                tags,
                self.next_token_tag(i),
                self.index + i
            );
        }
    }

    fn assert_tokens(&mut self, tags: &[TokenTag], offset: usize) {
        for tag in tags {
            if self.next_token_tag(offset) == *tag {
                return;
            }
        }
        unreachable!(
            "Error: expected tokens {:?}, got {:?}. Token index: {:?}.",
            tags,
            self.current_token_tag(),
            self.index
        );
    }

    fn expect_token(&mut self, tag: TokenTag) -> TokenId {
        self.assert_token(tag);
        self.shift_token()
    }

    fn expect_tokens(&mut self, tags: &[TokenTag]) -> TokenId {
        self.assert_tokens(tags, 0);
        self.shift_token()
    }

    fn expect_token_and_newline(&mut self, tag: TokenTag) -> TokenId {
        let token = self.expect_token(tag);
        self.match_token(TokenTag::Newline);
        token
    }

    fn token_is(&self, tag: TokenTag) -> bool {
        self.tree.tokens[self.index].tag == tag
    }

    fn token_isnt(&self, tag: TokenTag) -> bool {
        self.index < self.tree.tokens.len() - 1
            && !self.token_is(tag)
            && !self.token_is(TokenTag::Eof)
    }
}

const SPACES: usize = 2;

fn write_indent(f: &mut fmt::Formatter, indentation: usize) {
    write!(f, "{1:0$}", indentation * SPACES, "");
}

pub struct Tree {
    pub sources: Vec<String>,
    pub tokens: Vec<Token>,
    pub nodes: Vec<Node>,
    pub indices: Vec<u32>,
    pub module_indices: Vec<(String, usize)>,
}

impl Tree {
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
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

    pub fn node_lexeme(&self, id: NodeId) -> &str {
        let node = self.node(id);
        let token = self.node_token(node);
        let source = self.token_source(node.token);
        token.to_str(source)
    }

    pub fn node_lexeme_offset(&self, node: &Node, offset: i32) -> &str {
        if (node.token as i32 + offset) < 0 {
            panic!("bad offset")
        }
        let token = &self.tokens[(node.token as i32 + offset) as usize];
        let source = self.token_source(node.token);
        token.to_str(source)
    }

    fn token_source(&self, token_id: TokenId) -> &str {
        let mut source_index = 0;
        for (_, index) in &self.module_indices {
            if (token_id as usize) < *index {
                return &self.sources[source_index];
            }
            source_index += 1;
        }
        &self.sources[source_index]
    }

    /**************************************************************************/
    // Debug

    fn print_node(&self, f: &mut fmt::Formatter, id: NodeId, indentation: usize) {
        let node = self.node(id);
        let tag = node.tag;
        if node.lhs == 0 && node.rhs == 0 {
            if tag == Tag::Parameters {
                write!(f, "()");
            } else {
                write!(f, "{}", self.node_lexeme(id));
            }
            return;
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
                writeln!(f, "({:?}", tag);
                for i in node.lhs..node.rhs {
                    write_indent(f, indentation + 1);
                    self.print_node(f, self.node_index(i), indentation + 1);
                    writeln!(f, "");
                }
                write_indent(f, indentation);
                write!(f, ")");
            }
            // Multiple children, single line.
            Tag::Type => {
                write!(f, "({:?}", tag);
                for i in node.lhs..node.rhs {
                    write!(f, " ");
                    self.print_node(f, self.node_index(i), indentation + 1);
                }
                write!(f, ")");
            }
            // One or two children.
            _ => {
                write!(f, "({:?}", tag);
                if node.lhs != 0 {
                    write!(f, " ");
                    if tag == Tag::Field {
                        write!(f, "{} ", self.node_lexeme(id));
                    }
                    self.print_node(f, node.lhs, indentation);
                }
                if node.rhs != 0 && tag != Tag::Grouping {
                    write!(f, " ");
                    self.print_node(f, node.rhs, indentation);
                }
                write!(f, ")");
                return;
            }
        }
    }
}

impl fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let root = self.node(0);
        if root.lhs == 0 && root.rhs == 0 {
            self.print_node(f, (self.nodes.len() - 1) as u32, 0)
        } else {
            self.print_node(f, 0 as u32, 0)
        };
        Ok(())
    }
}
