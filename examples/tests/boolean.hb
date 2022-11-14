foo :: (condition: Bool, a: Int, b: Int) -> Int
    if condition
        return a
    else
        return b
    end
end

main :: () -> Int
    x := foo(!false && true || !true, 1, 3)
    y := foo(false || true && false, 4, 5)
    return x + y
end
