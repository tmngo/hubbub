main :: () -> Int
    return one()
end

#foreign-library "test_shared"
one :: () -> Int #foreign "hubbub-test-one"
