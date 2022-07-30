mutate :: (ptr: Pointer{Int}, x: Int)
    ptr@ = x
    return
end

Wrapper :: struct
    ptr: Pointer{Int}
end

main :: () -> Int
    value := 15
    print_int(value)

    p := &value
    p@ = 30
    print_int(p@)

    w: Wrapper
    w.ptr = p
    w.ptr@ = 45
    print_int(w.ptr@)

    mutate(p, 60)
    print_int(value)

    return 0
end
