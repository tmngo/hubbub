// Window :: Pointer{Void}
// Monitor :: Pointer{Int}

#foreign-library "SimpleDLL"

DisplayHelloFromMyDLL :: () #foreign


#foreign-library "glfw3"

#foreign-library "opengl32"

init :: () -> Int #foreign "glfwInit"

create-window :: (width: Int, height: Int, title: Pointer{u8}, monitor: Int, share: Int) -> Pointer{Int} #foreign "glfwCreateWindow"

get-key :: (window: Pointer{Int}, key: i32) -> i32 #foreign "glfwGetKey"

terminate :: () #foreign "glfwTerminate"

make-context-current :: (window: Pointer{Int}) #foreign "glfwMakeContextCurrent"

window-should-close :: (window: Pointer{Int}) -> Int #foreign "glfwWindowShouldClose"

set-window-should-close :: (window: Pointer{Int}, close: Bool) #foreign "glfwSetWindowShouldClose"

swap-buffers :: (window: Pointer{Int}) -> Int #foreign "glfwSwapBuffers"

poll-events :: () #foreign "glfwPollEvents"

gl-clear-color :: (red: f32, green: f32, blue: f32, alpha: f32) #foreign "glClearColor"

gl-clear :: (mask: u32) #foreign "glClear"
