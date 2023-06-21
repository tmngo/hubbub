#[cfg(test)]
pub const STRUCTS_TREE: &str = r#"
Module "<main>"
  Struct
    Field
      Identifier "x"
      Identifier "Int"
    Field
      Identifier "y"
      Identifier "Int"
  Struct
    Field
      Identifier "a"
      Identifier "Point"
    Field
      Identifier "b"
      Identifier "Point"
  FunctionDecl
    Prototype
      Parameters
      Expressions
        Identifier "Int"
    Block
      VariableDecl
        Expressions
          Identifier "segment"
        Identifier "Segment"
      Assign
        Access
          Access
            Identifier "segment"
            Identifier "a"
          Identifier "x"
        IntegerLiteral "3"
      VariableDecl
        Expressions
          Identifier "point"
        Identifier "Point"
      Assign
        Access
          Identifier "point"
          Identifier "y"
        IntegerLiteral "4"
      Return
        Expressions
          Add
            Access
              Access
                Identifier "segment"
                Identifier "a"
              Identifier "x"
            Access
              Identifier "point"
              Identifier "y""#;

#[cfg(test)]
pub const FIBONACCI_TREE: &str = r#"
Module "<main>"
  FunctionDecl
    Prototype
      Parameters
        Field
          Identifier "n"
          Identifier "Int"
      Expressions
        Identifier "Int"
    Block
      If
        Less
          Identifier "n"
          IntegerLiteral "2"
        Block
          Return
            Expressions
              Identifier "n"
      Return
        Expressions
          Add
            Call
              Identifier "fib"
              Expressions
                Sub
                  Identifier "n"
                  IntegerLiteral "1"
            Call
              Identifier "fib"
              Expressions
                Sub
                  Identifier "n"
                  IntegerLiteral "2"
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
            Identifier "fib"
            Expressions
              IntegerLiteral "7"
      Return
        Expressions
          Identifier "x""#;
