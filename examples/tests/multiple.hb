one-two :: () -> (Int, Int)
    return 1, 2
end

main :: () -> Int
    a, b := one-two()
    c, d, e := 3, 4 + b, 0
    return a + c + d + e
end
