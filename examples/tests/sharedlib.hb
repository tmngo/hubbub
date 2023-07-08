main :: () -> i32
    x := one() + addi64(3, 2)
    return i32(x)
end

#foreign-library "test_shared"
one :: () -> i64 #foreign "hubbub-test-one"

#foreign-library "test_shared"
addi64 :: (a: i64, b: i64) -> i64 #foreign "hubbub-test-addi64"
