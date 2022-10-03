f :: (a: Bool) -> Int
    if a
        return 1
    end
    return 0
end

f :: (a: Int) -> Int
    return a
end

+ :: #operator (c: Int, d: Bool) -> Int
    return 3
end

+ :: #operator (a: Bool, b: Bool) -> Int
    return 4
end

main :: () -> Int
    return f(true) + f(2) + (0 + true) + (true + true)
end
