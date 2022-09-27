Wrapper :: struct {T}
    arr: Array{T, 3}
    ptr: Pointer{T}
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
    return w.arr[0] + w.arr[1] + w.arr[2]
end
