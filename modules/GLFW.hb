// Window :: Pointer{Void}
// Monitor :: Pointer{Int}

#foreign-library "SimpleDLL"

DisplayHelloFromMyDLL :: () end


#foreign-library "glfw3"

glfwInit :: () -> Int end

glfwCreateWindow :: (width: Int, height: Int, title: Pointer{Int}, monitor: Int, share: Int) -> Pointer{Int} end

glfwTerminate :: () end

glfwMakeContextCurrent :: (window: Pointer{Int}) end

glfwWindowShouldClose :: (window: Pointer{Int}) -> Int end

glfwSwapBuffers :: (window: Pointer{Int}) -> Int end

glfwPollEvents :: () end
