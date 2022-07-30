use std::alloc::Layout;

pub extern "C" fn print_int(value: isize) -> isize {
    println!("{}", value);
    0
}

pub extern "C" fn alloc(n: isize) -> *mut i8 {
    println!("Allocate!");
    let layout = Layout::from_size_align(n as usize, 1).unwrap();
    unsafe { std::alloc::alloc(layout) as *mut i8 }
}

pub extern "C" fn dealloc(ptr: *mut i8, n: isize) {
    println!("Deallocate!");
    let layout = Layout::from_size_align(n as usize, 1).unwrap();
    unsafe {
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}
