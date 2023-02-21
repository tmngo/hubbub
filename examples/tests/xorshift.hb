main :: () -> Int
    y : i64 = 1
    y = xorshift(y)
    y = xorshift(y)
    y = xorshift(y)
    y = xorshift(y)
    y = xorshift(y)
    y = xorshift(y)
    return y
end

xorshift :: (c: i64) -> i64
    a : Int = c
    a = a << 7 ^ a
    a = a >> 9 ^ a
    a = a << 8 ^ a
    return a
end
