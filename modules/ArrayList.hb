#import "Base"

List :: struct {T}
    data: Pointer{T}
    length: Int
    capacity: Int
end

new-array-list :: (capacity: Int) -> List{Int}
    list: List{Int}
    list.data = alloc(capacity * 8)
    list.length = 0
    list.capacity = capacity
    return list
end

push :: (list: List{Int}, value: Int)
    return
end