#import "List"

Point :: struct
    x: Int
    y: Int
end

main :: () -> Int
    a: Point
    a.x = 3


    m := new-list(16)
    push(3, 3)
    return m.length
end
