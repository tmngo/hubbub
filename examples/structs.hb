// Math :: module "Math"

#import "Base"

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

point :: (x: Int, y: Int) -> Point
    point: Point
    point.x = x
    point.y = y
    return point
end

main :: () -> Int
    p := point(5, 1)
    q := point(5, 5)

    print_int(p.x)

    segment: Segment
    segment.a.x = 3
    segment.a.y = 4
    segment.b = q
    
    print_int(segment.a.x)
    print_int(segment.a.y)
    print_int(segment.b.x)

    print_int(magnitude(p))
    print_int(magnitude(segment.a))

    return 0
end
