// Math :: module "Math"

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
    p: Point
    p.x = 5
    p.y = 1

    q: Point
    q.x = 5 
    q.y = 5 
    
    segment: Segment
    segment.a.x = 3
    segment.a.y = 4
    segment.b = q
    
    print_int(segment.a.x)
    print_int(segment.a.y)
    print_int(segment.b.x)
    print_int(segment.b.y)

    print_int(magnitude(p))
    print_int(magnitude(segment.a))
    
    return 0
end
