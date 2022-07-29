mutate :: (ptr: Pointer{Int}, x: Int)
    ptr@ = x
    return
end

Wrapper :: struct
    ptr: Pointer{Int}
end

main :: () -> Int
    value := 10
    p := &value
    
    w: Wrapper
    w.ptr = p

    print_int(value)

    p@ = 30
    print_int(p@)

    p@ = 40
    print_int(w.ptr@)

    mutate(p, 50)
    print_int(value)

    return 0
end