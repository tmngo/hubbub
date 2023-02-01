exit :: (status: i32) #foreign

feof :: (f: Pointer{u8}) -> i32 #foreign

fgetc :: (f: Pointer{u8}) -> i32 #foreign

fopen :: (pathname: Pointer{u8}, mode: Pointer{u8}) -> Pointer{u8} #foreign

putchar :: (x: i32) -> i32 #foreign

puts :: (s: Pointer{u8}) -> i32 #foreign

// printf :: (s: Pointer{u8}) -> i32 #foreign

strlen :: (s: Pointer{u8}) -> i64 #foreign

