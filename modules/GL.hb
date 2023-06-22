#foreign-library "opengl32"

clear-color :: (red: f32, green: f32, blue: f32, alpha: f32) #foreign "glClearColor"

clear :: (mask: i32) #foreign "glClear"
