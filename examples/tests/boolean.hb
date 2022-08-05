foo :: (condition: Bool, a: Int, b: Int) -> Int
    if condition
        return a
    else
        return b
    end
end

main :: () -> Int
    x := foo(true, 1, 3)
    y := foo(false, 4, 5)
    return x + y
end
