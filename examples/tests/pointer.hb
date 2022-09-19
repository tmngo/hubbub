#import "Base"

mutate :: (ptr: Pointer{Int}, x: Int)
    ptr@ = x
    return
end

Wrapper :: struct
    ptr: Pointer{Int}
end

main :: () -> Int
    sum := 0
    value := 15
    sum = sum + value

    p := &value
    p@ = 30
    sum = sum + p@

    w: Wrapper
    w.ptr = p
    w.ptr@ = 45
    sum = sum + w.ptr@

    mutate(p, 60)
    sum = sum + value

    pointer := alloc(8)
    pointer@ = 8
    sum = sum + pointer@

    return sum
end
