// Window :: Pointer{Void}
// Monitor :: Pointer{Int}

#foreign-library "SimpleDLL"

DisplayHelloFromMyDLL :: () #foreign


#foreign-library "glfw3"

init :: () -> Int #foreign "glfwInit"

create-window :: (width: Int, height: Int, title: Pointer{Int}, monitor: Int, share: Int) -> Pointer{Int} #foreign "glfwCreateWindow"

terminate :: () #foreign "glfwTerminate"

make-context-current :: (window: Pointer{Int}) #foreign "glfwMakeContextCurrent"

window-should-close :: (window: Pointer{Int}) -> Int #foreign "glfwWindowShouldClose"

swap-buffers :: (window: Pointer{Int}) -> Int #foreign "glfwSwapBuffers"

poll-events :: () #foreign "glfwPollEvents"
