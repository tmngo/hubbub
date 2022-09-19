#import "Base"
#import "ArrayList"

A :: struct
    b: B
end

B :: struct
    c: C
end

C :: struct
    d: Int
end

main :: () -> Int
    ptr := alloc(2 * 8)
    print_int(ptr@)
    ptr2 := ptr
    print_int((ptr2)@)
    // x := 3
    // x = 4
    array: Array{Int, 4}
    i := 0
    while i < 4
        array[i] = 2 * i
        i = i + 1
    end
    while i > 0
        print_int(array[i - 1])
        i = i - 1
    end

    m := new-array-list(16)
    c := m.capacity
    // print_int(a)

    return m.capacity
end
