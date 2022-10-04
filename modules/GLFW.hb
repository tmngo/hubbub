// Window :: Pointer{Void}
// Monitor :: Pointer{Int}

glfwInit :: () -> Int end

glfwCreateWindow :: (width: Int, height: Int, title: Pointer{Int}, monitor: Int, share: Int) -> Pointer{Int} end

glfwTerminate :: () end

glfwMakeContextCurrent :: (window: Pointer{Int}) end

glfwWindowShouldClose :: (window: Pointer{Int}) -> Int end

glfwSwapBuffers :: (window: Pointer{Int}) -> Int end

glfwPollEvents :: () end
