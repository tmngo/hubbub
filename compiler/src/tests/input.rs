#[cfg(test)]
pub const STRUCTS_TREE: &str = r#"
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
      Identifier "a"
      Identifier "Point"
    Field
      Identifier "b"
      Identifier "Point"
  FunctionDecl
    Prototype
      Parameters
      Identifier "Int"
    Block
      VariableDecl
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
        Identifier "point"
        Identifier "Point"
      Assign
        Access
          Identifier "point"
          Identifier "y"
        IntegerLiteral "4"
      Return
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
Module
  FunctionDecl
    Prototype
      Parameters
        Field
          Identifier "n"
          Identifier "Int"
      Identifier "Int"
    Block
      If
        Less
          Identifier "n"
          IntegerLiteral "2"
        Block
          Return
            Identifier "n"
      Return
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
      Identifier "Int"
    Block
      VariableDecl
        Identifier "x"
        Call
          Identifier "fib"
          Expressions
            IntegerLiteral "7"
      Return
        Identifier "x""#;
