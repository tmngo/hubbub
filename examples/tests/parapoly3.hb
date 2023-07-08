zero :: {T}() -> T
    return 0
end

main :: () -> i32
    x := zero{i8}()
    y := zero{i32}()
    return x
end
