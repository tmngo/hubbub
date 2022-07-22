use std::alloc::Layout;

#[no_mangle]
fn print_int(i: isize) {
    println!("{}", i);
}

#[no_mangle]
fn alloc(n: isize) -> *mut i8 {
    let layout = Layout::from_size_align(n as usize, 1).unwrap();
    unsafe { std::alloc::alloc(layout) as *mut i8 }
}

#[no_mangle]
fn dealloc(ptr: *mut i8, n: isize) {
    let layout = Layout::from_size_align(n as usize, 1).unwrap();
    unsafe {
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        unsafe {
            let size = 2;
            let ptr = alloc(size) as *mut i16;
            *ptr = 4001;
            println!("{}", *(ptr as *mut i32));
            assert_eq!(*ptr, 4001);
            dealloc(ptr as *mut i8, size);
        }
    }
}
