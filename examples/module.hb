#import "Base"

// #import "Math"
Math2 :: #import "Math2"

main :: () -> Int
    // x := cube(3)
    x := Math2.cube(3)
    y := 1
    y = Math2.xorshift(y)
    print_int(y)
    y = Math2.xorshift(y)
    print_int(y)
    y = Math2.xorshift(y)
    print_int(y)
    y = Math2.xorshift(y)
    print_int(y)
    y = Math2.xorshift(y)
    print_int(y)
    y = Math2.xorshift(y)
    print_int(y)
    return x
end
