Wrapper :: struct {T}
    arr: Array{T, 3}
    ptr: Pointer{T}
end

Pair :: struct {A, B}
    a: A
    b: B
end

main :: () -> Int
    w: Wrapper{Int}
    z: Wrapper{Bool}
    condition := true
    z.ptr = &condition
    if z.ptr@
        w.arr[0] = 3
        w.arr[1] = 4
        w.arr[2] = 5
    end

    pair: Pair{Bool, Int}
    pair.a = false
    pair.b = 0 

    return w.arr[0] + w.arr[1] + w.arr[2] + pair.b
end
