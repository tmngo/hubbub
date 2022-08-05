#[cfg(test)]
pub const STRUCTS_TREE: &str = "\
(Root
  (Module
    (Struct
      (Field x Int)
      (Field y Int)
    )
    (Struct
      (Field a Point)
      (Field b Point)
    )
    (FunctionDecl (Prototype (Parameters
    ) Int) (Block
      (VariableDecl Segment)
      (Assign (Access (Access segment a) x) 3)
      (VariableDecl Point)
      (Assign (Access point y) 4)
      (Return (Add (Access (Access segment a) x) (Access point y)))
    ))
  )
)";

#[cfg(test)]
pub const FIBONACCI_TREE: &str = "\
(Root
  (Module
    (FunctionDecl (Prototype (Parameters
      (Field n Int)
    ) Int) (Block
      (If (Less n 2) (Block
        (Return n)
      ))
      (Return (Add (Call fib (Expressions
          (Sub n 1)
        )) (Call fib (Expressions
          (Sub n 2)
        ))))
    ))
    (FunctionDecl (Prototype (Parameters
    ) Int) (Block
      (VariableDecl (Call fib (Expressions
        7
      )))
      (Return x)
    ))
  )
)";
