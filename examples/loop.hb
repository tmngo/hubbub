main :: () -> Int64
    f := loop(5)
    return f
end

loop :: (k: Int) -> Int
    while k < 10
        print_int(k)
        k = k + 1
    end
    return k
end
