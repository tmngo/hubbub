main :: () -> Int
    y := 1
    y = xorshift(y)
    y = xorshift(y)
    y = xorshift(y)
    y = xorshift(y)
    y = xorshift(y)
    y = xorshift(y)
    return y
end

xorshift :: (a: Int) -> Int
    a = a << 7 ^ a
    a = a >> 9 ^ a
    a = a << 8 ^ a
    return a
end
