GLFW :: #import "GLFW"

main :: () -> Int
    GLFW.DisplayHelloFromMyDLL()

    x := GLFW.init()

    name := "hello"
    ptr := GLFW.create-window(640, 480, name.data, 0, 0)

    GLFW.make-context-current(ptr)
    while true
        GLFW.swap-buffers(ptr)
        GLFW.poll-events()
        if GLFW.window-should-close(ptr) > 0
            return 1
        end
    end

    return 0
end
