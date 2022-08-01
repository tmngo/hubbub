Point :: struct
    x: Int
    y: Int
end

Segment :: struct
    a: Point
    b: Point
end

magnitude :: (point: Point) -> Int
    return point.x + point.y
end

main :: () -> Int
    n: Int = 0
    n = 3
    b: Bool = true
    n = true
    n = 1 + false
    n = 1 + b
    
    return 0
end
