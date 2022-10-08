GLFW :: #import "GLFW"

main :: () -> Int
    GLFW.DisplayHelloFromMyDLL()

    x := GLFW.init()

    name := "hello"
    ptr := GLFW.create-window(640, 480, name.data, 0, 0)

    GLFW.make-context-current(ptr)
    while GLFW.window-should-close(ptr) == 0
        GLFW.swap-buffers(ptr)
        GLFW.poll-events()
    end

    return 0
end
