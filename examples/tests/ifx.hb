main :: () -> i32
    return ifx true
        => 1
    else
        => 2
    end
end

// if cond => 3 else => 5
// 
// if cond
//     => 3
// else
//     => 5 
// end
//
//
// if cond
//     => 2 // check the outer block
// end
//
// if cond
//     if cond2
//         => 1
//     end
//     => 2
// else
//     => 3
// end
//
// Parent if expressions:
// - call arguments
// - conversion arguments
// - assign rvalue
// - decl rvalue
// - return rvalue
// - => rvalue
// - if condition
// - while condition
//
// A parent if expression must:
// - have an else block
// - have a value for each block
// - matching types for each block
//
// A parent do expression must:
// - have a value for its block
