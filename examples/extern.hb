#import "Base"
#import "GLFW"

main :: () -> Int
    DisplayHelloFromMyDLL()

    x := glfwInit()

    name := 0
    ptr := glfwCreateWindow(640, 480, &name, 0, 0)

    glfwMakeContextCurrent(ptr)
    while true
        glfwSwapBuffers(ptr)
        glfwPollEvents()
        if glfwWindowShouldClose(ptr) > 0
            return 1
        end
    end

    return 0
end
