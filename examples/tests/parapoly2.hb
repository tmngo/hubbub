Counter :: struct {T}
    data: T
    count: i32
end

make-counter :: () -> Counter{Int}
    c: Counter{Int}
    c.count = i32(0)
    return c
end

update :: {T} (c: Pointer{Counter{T}}, data: T)
    c.data = data
    c.count = i32(c.count + 1)
    return
end

main :: () -> i32
    c := make-counter()
    update(&c, 19)
    return i32(c.data + c.count)
end
