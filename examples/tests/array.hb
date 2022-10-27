main :: () -> Int
    arr: Array{Int, 4}
    i := 1
    while i != 4
        arr[i] = 2 * i
        i = i + 1
    end
    return arr[1] + arr[3] - 8
end
