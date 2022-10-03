use phf::phf_map;
use std::{iter::Peekable, ops::Range, str::CharIndices};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Tag {
    Ampersand,
    AmpersandAmpersand,
    AmpersandEqual,
    Arrow,
    At,
    Bang,
    BangEqual,
    Block,
    BraceL,
    BraceR,
    BracketL,
    BracketR,
    Break,
    Caret,
    Colon,
    ColonColon,
    ColonEqual,
    Comma,
    Continue,
    Defer,
    Dollar,
    Dot,
    DotDot,
    DotDotDot,
    DotStar,
    DoubleArrow,
    Else,
    End,
    Enum,
    Eof,
    Equal,
    EqualEqual,
    False,
    Greater,
    GreaterEqual,
    GreaterGreater,
    Hash,
    Identifier,
    If,
    Import,
    IntegerLiteral,
    Invalid,
    Less,
    LessEqual,
    LessLess,
    Minus,
    MinusEqual,
    MinusMinus,
    Module,
    Newline,
    Operator,
    ParenL,
    ParenR,
    Percent,
    PercentEqual,
    PercentPercent,
    Pipe,
    PipeEqual,
    PipePipe,
    Plus,
    PlusEqual,
    PlusPlus,
    Return,
    Semicolon,
    SemicolonSemicolon,
    Slash,
    SlashEqual,
    Star,
    StarEqual,
    StarStar,
    StarStarStar,
    // StateError,
    StateIntegerLiteral10,
    // StateLineComment,
    StateLineCommentStart,
    StateStart,
    StateZero,
    StringLiteral,
    Struct,
    Tilde,
    True,
    // Typedef,
    // Union,
    Using,
    // Var,
    While,
}

static KEYWORDS: phf::Map<&'static str, Tag> = phf_map! {
    "if" => Tag::If,
    "end" => Tag::End,
    "else" => Tag::Else,
    "enum" => Tag::Enum,
    "true" => Tag::True,
    "block" => Tag::Block,
    "break" => Tag::Break,
    "defer" => Tag::Defer,
    "false" => Tag::False,
    "using" => Tag::Using,
    "while" => Tag::While,
    "module" => Tag::Module,
    "return" => Tag::Return,
    "struct" => Tag::Struct,
    "continue" => Tag::Continue,
};

static HASHTAGS: phf::Map<&'static str, Tag> = phf_map! {
    "import" => Tag::Import,
    "operator" => Tag::Operator,
};

#[derive(Copy, Clone, Debug)]
pub struct Token {
    pub tag: Tag,
    pub start: u32,
    pub end: u32,
}

impl Token {
    pub fn to_string(self, source: &str) -> String {
        source[self.start as usize..self.end as usize].to_string()
    }

    pub fn line(&self, source: &str) -> i64 {
        let mut newline_count = 1;
        for c in source[0..self.start as usize].chars() {
            if c == '\n' {
                newline_count += 1;
            }
        }
        newline_count
    }

    pub fn to_str(self, source: &str) -> &str {
        // println!("{} {}", self.start, self.end);
        if self.start as usize >= source.len() {
            return "EOF";
        }
        &source[self.start as usize..self.end as usize]
    }

    pub fn range(&self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

pub struct Tokenizer<'a> {
    index: u32,
    source: &'a str,
    char_indices: Peekable<CharIndices<'a>>,
    state: Tag,
    token: Token,
}

impl<'a> Tokenizer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            index: 0,
            source,
            state: Tag::StateStart,
            char_indices: source.char_indices().peekable(),
            token: Token {
                tag: Tag::Invalid,
                start: 0,
                end: 0,
            },
        }
    }

    ///
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens: Vec<Token> = Vec::new();
        loop {
            let token = self.next();
            tokens.push(token);
            if token.tag == Tag::Eof {
                return tokens;
            }
        }
    }

    pub fn append_tokens(&mut self, tokens: &mut Vec<Token>) {
        loop {
            let token = self.next();
            tokens.push(token);
            if token.tag == Tag::Eof {
                return;
            }
        }
    }

    ///
    pub fn next(&mut self) -> Token {
        let last = self.token.tag;
        self.token = Token {
            tag: Tag::Eof,
            start: self.index,
            end: self.index,
        };
        self.state = Tag::StateStart;
        loop {
            let c = if let Some((_, ch)) = self.char_indices.peek() {
                *ch
            } else {
                '\0'
            };
            match self.state {
                Tag::StateStart => {
                    self.token.start = self.index;
                    match c {
                        // End-of-file
                        '\0' => {
                            return self.token_fixed(Tag::Eof, 0);
                        }
                        // Whitespace
                        ' ' | '\r' => self.advance(),
                        '\n' => {
                            if last != Tag::Newline {
                                self.start(Tag::Newline);
                            } else {
                                self.advance();
                            }
                        }
                        // Identifier
                        'a'..='z' | 'A'..='Z' | '_' => self.start(Tag::Identifier),
                        // Unambiguous single-character tokens
                        '@' => return self.finish(Tag::At),
                        '{' => return self.finish(Tag::BraceL),
                        '}' => return self.finish(Tag::BraceR),
                        '[' => return self.finish(Tag::BracketL),
                        ']' => return self.finish(Tag::BracketR),
                        '^' => return self.finish(Tag::Caret),
                        ',' => return self.finish(Tag::Comma),
                        '$' => return self.finish(Tag::Dollar),
                        '(' => return self.finish(Tag::ParenL),
                        ')' => return self.finish(Tag::ParenR),
                        '~' => return self.finish(Tag::Tilde),
                        // Potential multi-character tokens
                        '&' => self.start(Tag::Ampersand),
                        '!' => self.start(Tag::Bang),
                        ':' => self.start(Tag::Colon),
                        '.' => self.start(Tag::Dot),
                        '=' => self.start(Tag::Equal),
                        '>' => self.start(Tag::Greater),
                        '#' => self.start(Tag::Hash),
                        '<' => self.start(Tag::Less),
                        '-' => self.start(Tag::Minus),
                        '%' => self.start(Tag::Percent),
                        '|' => self.start(Tag::Pipe),
                        '+' => self.start(Tag::Plus),
                        ';' => self.start(Tag::Semicolon),
                        '/' => self.start(Tag::Slash),
                        '*' => self.start(Tag::Star),
                        // String literals
                        '"' => self.start(Tag::StringLiteral),
                        // Integer literals
                        '0' => self.start(Tag::StateZero),
                        '1'..='9' => self.start(Tag::StateIntegerLiteral10),
                        _ if unicode_ident::is_xid_start(c) => self.start(Tag::Identifier),
                        // Invalid tokens
                        _ => {
                            panic!("invalid token {:?}", c);
                        }
                    }
                }
                Tag::Ampersand => match c {
                    '&' => return self.finish(Tag::AmpersandAmpersand),
                    '=' => return self.finish(Tag::AmpersandEqual),
                    _ => return self.token(Tag::Ampersand),
                },
                Tag::Bang => match c {
                    '=' => return self.finish(Tag::BangEqual),
                    _ => return self.token(Tag::Bang),
                },
                Tag::Colon => match c {
                    ':' => return self.finish(Tag::ColonColon),
                    '=' => return self.finish(Tag::ColonEqual),
                    _ => return self.token(Tag::Colon),
                },
                Tag::Dot => match c {
                    '.' => self.start(Tag::DotDot),
                    '*' => return self.finish(Tag::DotStar),
                    _ => return self.token(Tag::Dot),
                },
                Tag::DotDot => match c {
                    '.' => return self.finish(Tag::DotDotDot),
                    _ => return self.token(Tag::DotDot),
                },
                Tag::Equal => match c {
                    '>' => return self.finish(Tag::DoubleArrow),
                    '=' => return self.finish(Tag::EqualEqual),
                    _ => return self.token(Tag::Equal),
                },
                Tag::Greater => match c {
                    '=' => return self.finish(Tag::GreaterEqual),
                    '>' => return self.finish(Tag::GreaterGreater),
                    _ => return self.token(Tag::Greater),
                },
                Tag::Hash => match c {
                    'a'..='z' | '-' => self.advance(),
                    _ => return self.hashtag(),
                },
                Tag::Identifier => match c {
                    'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => self.advance(),
                    _ if unicode_ident::is_xid_continue(c) => self.advance(),
                    _ => return self.keyword_or_identifier(),
                },
                Tag::Less => match c {
                    '=' => return self.finish(Tag::LessEqual),
                    '<' => return self.finish(Tag::LessLess),
                    _ => return self.token(Tag::Less),
                },
                Tag::Minus => match c {
                    '>' => return self.finish(Tag::Arrow),
                    '=' => return self.finish(Tag::MinusEqual),
                    '-' => return self.finish(Tag::MinusMinus),
                    _ => return self.token(Tag::Minus),
                },
                Tag::Newline => match c {
                    ' ' | '\n' | '\r' | '\t' => self.advance(),
                    _ => return self.token_fixed(Tag::Newline, 1),
                },
                Tag::Percent => match c {
                    '=' => return self.finish(Tag::PercentEqual),
                    '%' => return self.finish(Tag::PercentPercent),
                    _ => return self.token(Tag::Percent),
                },
                Tag::Pipe => match c {
                    '=' => return self.finish(Tag::PipeEqual),
                    '|' => return self.finish(Tag::PipePipe),
                    _ => return self.token(Tag::Pipe),
                },
                Tag::Plus => match c {
                    '=' => return self.finish(Tag::PlusEqual),
                    '+' => return self.finish(Tag::PlusPlus),
                    _ => return self.token(Tag::Plus),
                },
                Tag::Semicolon => match c {
                    ';' => return self.finish(Tag::SemicolonSemicolon),
                    _ => return self.token(Tag::Semicolon),
                },
                Tag::Slash => match c {
                    '/' => self.start(Tag::StateLineCommentStart),
                    '=' => return self.finish(Tag::SlashEqual),
                    _ => return self.token(Tag::Slash),
                },
                Tag::Star => match c {
                    '=' => return self.finish(Tag::StarEqual),
                    '*' => self.start(Tag::StarStar),
                    _ => return self.token(Tag::Star),
                },
                Tag::StarStar => match c {
                    '*' => return self.finish(Tag::StarStarStar),
                    _ => return self.token(Tag::StarStar),
                },
                Tag::StringLiteral => match c {
                    '"' => return self.finish(Tag::StringLiteral),
                    _ => self.advance(),
                },
                Tag::StateLineCommentStart => match c {
                    '\n' => {
                        if last == Tag::Newline {
                            self.start(Tag::StateStart);
                        } else {
                            self.start(Tag::Newline);
                        }
                    }
                    '\0' => return self.token_fixed(Tag::Eof, 0),
                    _ => self.advance(),
                },
                Tag::StateIntegerLiteral10 => match c {
                    '0'..='9' | '_' => self.advance(),
                    _ => return self.token(Tag::IntegerLiteral),
                },
                Tag::StateZero => match c {
                    '0'..='9' | '_' => self.start(Tag::StateIntegerLiteral10),
                    _ => return self.token(Tag::IntegerLiteral),
                },
                _ => {
                    unreachable!("invalid tokenizer state")
                }
            }
        }
    }

    fn advance(&mut self) {
        self.char_indices.next();
        if let Some(char_index) = self.char_indices.peek() {
            self.index = char_index.0 as u32;
        } else {
            self.index = self.source.len() as u32;
        }
    }

    fn start(&mut self, tag: Tag) {
        self.token.tag = tag;
        self.state = tag;
        self.advance();
    }

    fn finish(&mut self, tag: Tag) -> Token {
        self.advance();
        self.token(tag)
    }

    fn token(&mut self, tag: Tag) -> Token {
        self.token.tag = tag;
        self.token.end = self.index;
        self.token
    }

    fn token_fixed(&mut self, tag: Tag, length: u32) -> Token {
        self.token.tag = tag;
        self.token.end = self.token.start + length;
        self.token
    }

    fn keyword_or_identifier(&mut self) -> Token {
        match KEYWORDS.get(&self.source[self.token.start as usize..self.index as usize]) {
            Some(tag) => self.token(*tag),
            None => self.token(Tag::Identifier),
        }
    }

    fn hashtag(&mut self) -> Token {
        match HASHTAGS.get(&self.source[(self.token.start + 1) as usize..self.index as usize]) {
            Some(tag) => self.token(*tag),
            None => unreachable!("invalid hashtag"),
        }
    }
}

pub fn print(source: &str, tokens: &[Token]) {
    for (i, token) in tokens.iter().enumerate() {
        println!(
            "{:<5} {:<16} {:<9} {}",
            i,
            format!("{:?}", token.tag),
            format!("{}..{}", token.start, token.end),
            token.to_str(source).replace('\n', "\\n")
        );
    }
}
