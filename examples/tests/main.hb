square :: (a: Int) -> Int
    return a * a
end

f :: (a: Int, b: Int) -> Int
    c := 2 * (a + b)
    return c
end

main :: () -> Int64
    a := 3
    b := 5
    c := f(a, b) 
    d := square(b) 
    e := 0
    if d < 24
        e = 10
    else if d < 26
        e = 20
    else if d < 30
        e = 30
    end
    return c + d + e
end
