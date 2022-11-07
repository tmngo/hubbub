use crate::{
    link::set_default_absolute_module_path, parse::Parser, tokenize::Tokenizer,
    workspace::Workspace,
};

pub enum Test {
    File,
    Expr,
    Stmt,
}

pub fn test_parse(test: Test, source: &str, expected: &str) {
    set_default_absolute_module_path();
    let mut tokenizer = Tokenizer::new(source);
    let tokens = tokenizer.tokenize();

    let mut workspace = Workspace::new();

    let mut parser = Parser::new(&mut workspace, source, tokens);

    match test {
        Test::File => {
            parser.parse();
        }
        Test::Expr => {
            parser.parse_expr().ok();
        }
        Test::Stmt => {
            parser.parse_stmt().ok();
        }
    };

    let tree = parser.tree();
    let result = format!("{:?}", tree);
    assert_eq!(
        &result, expected,
        "expected:\n{}\n\ngot:\n{}\n",
        expected, &result
    );
}

#[test]
fn parse_simple() {
    test_parse(
        Test::File,
        "\
f :: (a: Int, b: Int) -> Int
    c := 2 * (a + b)
    return c
end
        ",
        r#"
Module
  FunctionDecl
    Prototype
      Parameters
        Field
          Identifier "a"
          Identifier "Int"
        Field
          Identifier "b"
          Identifier "Int"
      Expressions
        Identifier "Int"
    Block
      VariableDecl
        Expressions
          Identifier "c"
        Expressions
          Mul
            IntegerLiteral "2"
            Add
              Identifier "a"
              Identifier "b"
      Return
        Identifier "c""#,
    );
}

#[test]
fn parse_comments() {
    test_parse(
        Test::File,
        "\
// Comment 1.
main :: () -> Int 
    // Comment 2.
    x := 3 // End of line comment.
    // Comment 3.
    // Comment 4.
    // Comment 5.
    return x
end // End of line comment.
// Comment 6.
",
        r#"
Module
  FunctionDecl
    Prototype
      Parameters
      Expressions
        Identifier "Int"
    Block
      VariableDecl
        Expressions
          Identifier "x"
        Expressions
          IntegerLiteral "3"
      Return
        Identifier "x""#,
    );
}

#[test]
fn parse_expressions() {
    test_parse(
        Test::Expr,
        "a + b + c",
        r#"
Add
  Add
    Identifier "a"
    Identifier "b"
  Identifier "c""#,
    );
    test_parse(
        Test::Expr,
        "- - -x",
        r#"
Negation
  Negation
    Negation
      Identifier "x""#,
    );
    test_parse(
        Test::Expr,
        "a + (b * 4) + 2",
        r#"
Add
  Add
    Identifier "a"
    Mul
      Identifier "b"
      IntegerLiteral "4"
  IntegerLiteral "2""#,
    );
    test_parse(
        Test::Expr,
        "a > b || c < d",
        r#"
LogicalOr
  Greater
    Identifier "a"
    Identifier "b"
  Less
    Identifier "c"
    Identifier "d""#,
    );
    test_parse(
        Test::Expr,
        "a + b * (4 + 2)",
        r#"
Add
  Identifier "a"
  Mul
    Identifier "b"
    Add
      IntegerLiteral "4"
      IntegerLiteral "2""#,
    );
    test_parse(
        Test::Expr,
        "- -1 * 2",
        r#"
Mul
  Negation
    Negation
      IntegerLiteral "1"
  IntegerLiteral "2""#,
    );
    test_parse(
        Test::Expr,
        "(((0)))",
        r#"
IntegerLiteral "0""#,
    );

    test_parse(
        Test::Expr,
        "-9!!",
        r#"
Negation
  Factorial
    Factorial
      IntegerLiteral "9""#,
    );
    test_parse(
        Test::Expr,
        "(-9!)!",
        r#"
Factorial
  Negation
    Factorial
      IntegerLiteral "9""#,
    );
    test_parse(
        Test::Expr,
        "arr[0][1]",
        r#"
Subscript
  Subscript
    Identifier "arr"
    IntegerLiteral "0"
  IntegerLiteral "1""#,
    );
    test_parse(
        Test::Expr,
        "x.y.arr[0]",
        r#"
Subscript
  Access
    Access
      Identifier "x"
      Identifier "y"
    Identifier "arr"
  IntegerLiteral "0""#,
    );
    test_parse(
        Test::Expr,
        "&p.x",
        r#"
Address
  Access
    Identifier "p"
    Identifier "x""#,
    );
}

#[test]
fn parse_types() {
    test_parse(
        Test::File,
        "\
func :: (a: Int64, b: Pointer{Int64}) -> (Int64, Array{Int64})
    c: Flo64
    d: Array{Array{Int64}}
    return 0
end",
        r#"
Module
  FunctionDecl
    Prototype
      Parameters
        Field
          Identifier "a"
          Identifier "Int64"
        Field
          Identifier "b"
          Type "Pointer"
            Identifier "Int64"
      Expressions
        Identifier "Int64"
        Type "Array"
          Identifier "Int64"
    Block
      VariableDecl
        Expressions
          Identifier "c"
        Identifier "Flo64"
      VariableDecl
        Expressions
          Identifier "d"
        Type "Array"
          Type "Array"
            Identifier "Int64"
      Return
        IntegerLiteral "0""#,
    )
}

#[test]
fn parse_assign() {
    test_parse(
        Test::Stmt,
        "a := b;",
        r#"
VariableDecl
  Expressions
    Identifier "a"
  Expressions
    Identifier "b""#,
    );
}

#[test]
fn parse_module() {
    test_parse(
        Test::File,
        "\
Math :: #import \"Math\"
main :: () -> Int
    x := Math.cube(3)
    return 0
end\
      ",
        r#"
Module
  Import
    Identifier "Math"
    StringLiteral ""Math""
  FunctionDecl
    Prototype
      Parameters
      Expressions
        Identifier "Int"
    Block
      VariableDecl
        Expressions
          Identifier "x"
        Expressions
          Call
            Access
              Identifier "Math"
              Identifier "cube"
            Expressions
              IntegerLiteral "3"
      Return
        IntegerLiteral "0"
Module
  FunctionDecl
    Prototype
      Parameters
        Field
          Identifier "a"
          Identifier "Int"
      Expressions
        Identifier "Int"
    Block
      Return
        Mul
          Mul
            Identifier "a"
            Identifier "a"
          Identifier "a""#,
    )
}

#[test]
fn parse_variables() {
    test_parse(
        Test::File,
        "\
main :: ()
    a: Int8
    b := 1 + 2
    c: Int64 = b
    a = 0
    return 0
end",
        r#"
Module
  FunctionDecl
    Prototype
      Parameters
    Block
      VariableDecl
        Expressions
          Identifier "a"
        Identifier "Int8"
      VariableDecl
        Expressions
          Identifier "b"
        Expressions
          Add
            IntegerLiteral "1"
            IntegerLiteral "2"
      VariableDecl
        Expressions
          Identifier "c"
        Identifier "Int64"
        Expressions
          Identifier "b"
      Assign
        Identifier "a"
        IntegerLiteral "0"
      Return
        IntegerLiteral "0""#,
    );
}

#[test]
fn parse_while() {
    test_parse(
        Test::File,
        "\
main :: () -> Int64
    k := 0
    while k < 10
        putchar(65)
        k = k + 1
    end
    return k
end
    ",
        r#"
Module
  FunctionDecl
    Prototype
      Parameters
      Expressions
        Identifier "Int64"
    Block
      VariableDecl
        Expressions
          Identifier "k"
        Expressions
          IntegerLiteral "0"
      While
        Less
          Identifier "k"
          IntegerLiteral "10"
        Block
          Call
            Identifier "putchar"
            Expressions
              IntegerLiteral "65"
          Assign
            Identifier "k"
            Add
              Identifier "k"
              IntegerLiteral "1"
      Return
        Identifier "k""#,
    )
}

#[test]
fn parse_struct() {
    test_parse(
        Test::File,
        "\
Point :: struct
    x: Int
    y: Int
end

Vector3 :: struct
    x: Float
    y: Float
    z: Float
end",
        r#"
Module
  Struct
    Field
      Identifier "x"
      Identifier "Int"
    Field
      Identifier "y"
      Identifier "Int"
  Struct
    Field
      Identifier "x"
      Identifier "Float"
    Field
      Identifier "y"
      Identifier "Float"
    Field
      Identifier "z"
      Identifier "Float""#,
    );
}

#[test]
fn parse_if_stmt() {
    test_parse(
        Test::Stmt,
        "\
if x > 90
    a
end",
        r#"
If
  Greater
    Identifier "x"
    IntegerLiteral "90"
  Block
    Identifier "a""#,
    );
}

#[test]
fn parse_if_else() {
    test_parse(
        Test::Stmt,
        "\
if x > 90
    if x > 95
        ap
    end
    a
else if x > 80
    b
else if x > 70
    c
else if x > 60
    d
else
    f
end",
        r#"
IfElse
  If
    Greater
      Identifier "x"
      IntegerLiteral "90"
    Block
      If
        Greater
          Identifier "x"
          IntegerLiteral "95"
        Block
          Identifier "ap"
      Identifier "a"
  If
    Greater
      Identifier "x"
      IntegerLiteral "80"
    Block
      Identifier "b"
  If
    Greater
      Identifier "x"
      IntegerLiteral "70"
    Block
      Identifier "c"
  If
    Greater
      Identifier "x"
      IntegerLiteral "60"
    Block
      Identifier "d"
  If
    Block
      Identifier "f""#,
    );
}

#[test]
fn parse_parametric_fn() {
    test_parse(
        Test::File,
        "\
equals :: {T} (a: T, b: T) -> Int
    if a == b
        return 1
    else
        return 0
    end
end
    ",
        r#"
Module
  FunctionDecl
    Prototype
      TypeParameters
        TypeParameter
      Parameters
        Field
          Identifier "a"
          Identifier "T"
        Field
          Identifier "b"
          Identifier "T"
      Expressions
        Identifier "Int"
    Block
      IfElse
        If
          Equality
            Identifier "a"
            Identifier "b"
          Block
            Return
              IntegerLiteral "1"
        If
          Block
            Return
              IntegerLiteral "0""#,
    )
}
