GLFW :: #import "GLFW"
GL :: #import "GL"

main :: () -> Int
    x := GLFW.init()

    name := "GLFW Window"
    window := GLFW.create-window(640, 480, name.data, 0, 0)

    GLFW.make-context-current(window)
    GL.clear-color(1.0, 0.0, 0.5, 1.0)

    // Render loop
    while GLFW.window-should-close(window) == 0
        // Input
        process-input(window)

        // Render
        GL.clear(16384)

        // Swap buffers and handle events
        GLFW.swap-buffers(window)
        GLFW.poll-events()
    end

    return 0
end

process-input :: (window: Pointer{Int})
    GLFW-KEY-ESCAPE : i32 = 256
    GLFW-PRESS : i32 = 1
    if GLFW.get-key(window, GLFW-KEY-ESCAPE) == GLFW-PRESS
        GLFW.set-window-should-close(window, true)
    end
    return
end
