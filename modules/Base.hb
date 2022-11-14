C :: #import "C"

alloc :: (n: Int) -> Pointer{Int} end

print_int :: (n: Int) end

print_f32 :: (n: f32) end

print_f64 :: (n: f64) end

assert :: (condition: Bool)
    if condition == false
        C.exit(1)
    end
    return
end

exit :: (status: Int)
    C.exit(i32(status))
    return
end
