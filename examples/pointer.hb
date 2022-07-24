mutate :: (ptr: Pointer{Int}, x: Int) -> Int
    @ptr = x
    return 0
end

main :: () -> Int
    x := 6
    p := &x
    print_int(x)
    print_int(p)
    print_int(@p)
    @p = 3
    print_int(x)
    mutate(p, 5)
    print_int(x)
    return 0
end