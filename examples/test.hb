Base :: #import "Base"
C :: #import "C"

main :: () -> Int
    x := 3
    y := "hello"
    z := C.strlen(y.data)
    pathname := "test.hb"
    mode := "r"
    a : i32 = 3
    f := C.fopen(pathname.data, mode.data)
    Base.print_cstr(pathname.data)
    c := C.fgetc(f)
    while c != i32(-1)
        C.putchar((c))
        c = C.fgetc(f)
    end
    Base.print_int(z)
    return 0
end
