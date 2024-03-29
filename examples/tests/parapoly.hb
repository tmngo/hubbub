identity :: {T}(x: T) -> T
    return x
end

three :: {T}(x: T) -> Int
    return 3
end

deref :: {T}(p: Pointer{T}) -> T
    return p@
end

mutate :: {T}(p: Pointer{T}, x: T)
    p@ = x
    return
end

main :: () -> i32
    x := identity(3)
    y := 0
    z := false
    ptr := &z
    mutate(ptr, identity(true))
    c := deref(ptr)
    if c
        mutate(&y, -three(false))
    end
    return i32(x + y)
end
