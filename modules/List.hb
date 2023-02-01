Base :: #import "Base"

List :: struct {T}
    data: Pointer{T}
    length: Int
    capacity: Int
end

new-list :: (capacity: Int) -> List{Int}
    list: List{Int}
    list.data = Base.alloc(capacity * 8)
    list.length = 0
    list.capacity = capacity
    return list
end

push :: {T} (list: Pointer{List{T}}, value: T)
    list.length = list.length + 1
    msg := "push"
    Base.print_cstr(msg.data)
    return
end
