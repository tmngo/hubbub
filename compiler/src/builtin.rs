use std::alloc::Layout;

pub extern "C" fn print_int(value: isize) -> isize {
    println!("{value}");
    0
}

pub extern "C" fn print_f32(value: f32) -> isize {
    println!("{value}");
    0
}

pub extern "C" fn print_f64(value: f64) -> isize {
    println!("{value}");
    0
}

pub extern "C" fn print_cstr(s: *const i8) {
    let s = unsafe { std::ffi::CStr::from_ptr(s) };
    println!("{}", s.to_str().unwrap());
}

pub extern "C" fn alloc(n: isize) -> *mut i8 {
    println!("builtin.rs:alloc(n={})", n);
    let layout = Layout::from_size_align(n as usize, 1).unwrap();
    unsafe { std::alloc::alloc(layout) as *mut i8 }
}

pub extern "C" fn dealloc(ptr: *mut i8, n: isize) {
    println!("builtin.rs:dealloc(ptr={:p}, n={})", ptr, n);
    let layout = Layout::from_size_align(n as usize, 1).unwrap();
    unsafe {
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}
