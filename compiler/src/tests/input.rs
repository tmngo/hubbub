#[cfg(test)]
pub const STRUCTS: &str = "\
Point :: struct
    x: Int
    y: Int
end
    
Segment :: struct
    a: Point
    b: Point
end

main :: () -> Int
    segment: Segment
    segment.a.x = 3
    point: Point
    point.y = 4
    return segment.a.x + point.y
end
";

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
pub const FIBONACCI: &str = "\
fib :: (n: Int) -> Int
    if n < 2
        return n
    end
    return fib(n - 1) + fib(n - 2)
end

main :: () -> Int
    x := fib(7)
    print_int(x)
    return x
end
";

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
      (Call print_int (Expressions
        x
      ))
      (Return x)
    ))
  )
)";
