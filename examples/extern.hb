GLFW :: #import "GLFW"
GL :: #import "GL"

main :: () -> Int
    GLFW.DisplayHelloFromMyDLL()

    x := GLFW.init()

    name := "hello"
    ptr := GLFW.create-window(640, 480, name.data, 0, 0)

    GLFW.make-context-current(ptr)
    GL.clear-color(1.0, 0.0, 0.5, 0.0)

    while GLFW.window-should-close(ptr) == 0
        GL.clear(16384)
        GLFW.swap-buffers(ptr)
        GLFW.poll-events()
    end

    return 0
end
