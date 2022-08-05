Point :: struct
    x: Int
    y: Int
end
    
Segment :: struct
    a: Point
    b: Point
end

main :: () -> Int
    segment: Segment
    segment.a.x = 3
    point: Point
    point.y = 4
    return segment.a.x + point.y
end