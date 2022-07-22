fib :: (n: Int) -> Int
    if n < 2
        return n
    end
    return fib(n - 1) + fib(n - 2)
end

main :: () -> Int
    x := fib(7)
    print_int(x)
    return x
end
