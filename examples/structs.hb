// Math :: module "Math"

Point :: struct
    x: Int
    y: Int
end

magnitude :: (point: Point) -> Int
    return point.x + point.y
end

main :: () -> Int
    point: Point
    point.x = 3
    point.y = 4
    print_int(point.x)
    print_int(point.y)
    print_int(magnitude(point))
    return point.x + point.y
end