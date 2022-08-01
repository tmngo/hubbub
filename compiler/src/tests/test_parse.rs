use crate::analyze::Analyzer;
use crate::parse::Parser;
use crate::tests::input::*;
use crate::tokenize::Tokenizer;

pub enum Test {
  File,
  Expr,
  Stmt,
}

pub fn test_parse(test: Test, src: &str, expected: &str) {
  let source = src.to_string();

  // Tokenize
  let mut tokenizer = Tokenizer::new(&source);
  let tokens = tokenizer.tokenize();

  // Parse
  let mut parser = Parser::new(&source, tokens);

  match test {
    Test::File => {
      parser.parse().ok();
    }
    Test::Expr => {
      parser.parse_expr().ok();
    }
    Test::Stmt => {
      parser.parse_stmt().ok();
    }
  };

  let tree = parser.tree();
  println!("Nodes: {}", tree.nodes.len());
  let result = format!("{}", tree);
  println!("{}", result);
  assert_eq!(&result, expected);
}

pub fn test_analyze(src: &str, def_count: usize) {
  let source = src.to_string();
  // source.push('\0');

  // Tokenize
  let mut tokenizer = Tokenizer::new(&source);
  let tokens = tokenizer.tokenize();

  // Parse
  let mut parser = Parser::new(&source, tokens);
  parser.parse().ok();
  let tree = parser.tree();
  println!("{}", tree);

  // Analyze
  let mut analyzer = Analyzer::new(&tree);
  analyzer.resolve().ok();
  assert_eq!(def_count, analyzer.definitions.len());
  println!("{}", analyzer);
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
    "\
(Root
  (Module
    (FunctionDecl (Prototype (Parameters
      (Field a Int)
      (Field b Int)
    ) Int) (Block
      (VariableDecl (Mul 2 (Add a b)))
      (Return c)
    ))
  )
)",
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
    "\
(Root
  (Module
    (FunctionDecl (Prototype () Int) (Block
      (VariableDecl 3)
      (Return x)
    ))
  )
)",
  );
}

#[test]
fn parse_expressions() {
  test_parse(Test::Expr, "a + b + c", "(Add (Add a b) c)");
  test_parse(Test::Expr, "- - -x", "(Negation (Negation (Negation x)))");
  test_parse(Test::Expr, "a + (b * 4) + 2", "(Add (Add a (Mul b 4)) 2)");
  test_parse(
    Test::Expr,
    "a > b || c < d",
    "(LogicalOr (Greater a b) (Less c d))",
  );
  test_parse(Test::Expr, "a + b * (4 + 2)", "(Add a (Mul b (Add 4 2)))");
  test_parse(Test::Expr, "- -1 * 2", "(Mul (Negation (Negation 1)) 2)");
  test_parse(Test::Expr, "(((0)))", "0");

  test_parse(Test::Expr, "-9!!", "(Negation (Factorial (Factorial 9)))");
  test_parse(Test::Expr, "(-9!)!", "(Factorial (Negation (Factorial 9)))");
  test_parse(Test::Expr, "arr[0][1]", "(Subscript (Subscript arr 0) 1)");
  test_parse(
    Test::Expr,
    "x.y.arr[0]",
    "(Subscript (Access (Access x y) arr) 0)",
  );
  test_parse(Test::Expr, "&p.x", "(Address (Access p x))");
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
    "\
(Root
  (Module
    (FunctionDecl (Prototype (Parameters
      (Field a Int64)
      (Field b (Type Int64))
    ) (Expressions
      Int64
      (Type Int64)
    )) (Block
      (VariableDecl Flo64)
      (VariableDecl (Type (Type Int64)))
      (Return 0)
    ))
  )
)",
  )
}

#[test]
fn parse_assign() {
  test_parse(Test::Stmt, "a := b;", "(VariableDecl b)");
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
    "\
(Root
  (Module
    (Import Math \"Math\")
    (FunctionDecl (Prototype () Int) (Block
      (VariableDecl (Call (Access Math cube) (Expressions
        3
      )))
      (Return 0)
    ))
  )
  (Module
    (FunctionDecl (Prototype (Parameters
      (Field a Int)
    ) Int) (Block
      (Return (Mul (Mul a a) a))
    ))
  )
)\
      ",
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
    "\
(Root
  (Module
    (FunctionDecl (Prototype ()) (Block
      (VariableDecl Int8)
      (VariableDecl (Add 1 2))
      (VariableDecl Int64 b)
      (Assign a 0)
      (Return 0)
    ))
  )
)",
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
    "\
(Root
  (Module
    (FunctionDecl (Prototype () Int64) (Block
      (VariableDecl 0)
      (While (Less k 10) (Block
        (Call putchar (Expressions
          65
        ))
        (Assign k (Add k 1))
      ))
      (Return k)
    ))
  )
)",
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
    "\
(Root
  (Module
    (Struct
      (Field x Int)
      (Field y Int)
    )
    (Struct
      (Field x Float)
      (Field y Float)
      (Field z Float)
    )
  )
)",
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
    "\
(If (Greater x 90) (Block
  a
))",
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
    "\
(IfElse
  (If (Greater x 90) (Block
    (If (Greater x 95) (Block
      ap
    ))
    a
  ))
  (If (Greater x 80) (Block
    b
  ))
  (If (Greater x 70) (Block
    c
  ))
  (If (Greater x 60) (Block
    d
  ))
  (If (Block
    f
  ))
)",
  );
}
#[test]
fn analyze_simple() {
  test_analyze(
    "\
    i: Int = 9
    j: Int
    // i: Int
    f :: (x: Int, y: Float) -> Float
        i: Float
        block
            z: Float = x + y
            i = z
        end
        block
            z: Float = i + 1
            i = z
        end
        return i
    end
    g :: ()
        f(1, 2)
    end
        ",
    17,
  );
}

#[test]
fn analyze_struct() {
  test_analyze(STRUCTS, 23);
}
