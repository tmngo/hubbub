f :: (a: Bool) -> Int
    if a
        return 1
    end
    return 0
end

f :: (a: Int) -> Int
    return a
end

main :: () -> Int
    return f(true) + f(2)
end
