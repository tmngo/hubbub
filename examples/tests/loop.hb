main :: () -> Int
    return loop(5, 10)
end

loop :: (a: Int, b: Int) -> Int
    sum := 0
    while a < b
        sum = sum + a
        a = a + 1
    end
    return sum
end
