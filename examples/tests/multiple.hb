one-two :: () -> (Int, Int)
    return 1, 2
end

main :: () -> Int
    a, b := one-two()
    c, d, e := 3, 4 + b, 0
    f, g, h := one-two(), 0
    i, j, k := 4, one-two()
    return a + c + d + e + h + j - 1
end
