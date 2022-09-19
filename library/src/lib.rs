use std::alloc;

#[export_name = "Base.print_int"]
fn print_int(i: isize) {
    println!("{}", i);
}

#[export_name = "Base.alloc"]
fn alloc(n: isize) -> *mut i8 {
    println!("lib.rs:alloc(n={})", n);
    let layout = alloc::Layout::from_size_align(n as usize, 1).unwrap();
    unsafe { alloc::alloc(layout) as *mut i8 }
}

#[no_mangle]
fn dealloc(ptr: *mut i8, n: isize) {
    println!("lib.rs:dealloc(ptr={:p}, n={})", ptr, n);
    let layout = alloc::Layout::from_size_align(n as usize, 1).unwrap();
    unsafe {
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

const ALIGN: usize = 8;
const EXTRA_SIZE: usize = std::mem::size_of::<usize>();

#[no_mangle]
fn alloc_implicit(count: i64, bytes: i64) -> *mut u8 {
    // The logical size of the allocation.
    let payload_size: usize = (count * bytes).try_into().unwrap();

    // Allocate one extra word to store the size.
    let layout = alloc::Layout::from_size_align(payload_size + EXTRA_SIZE, ALIGN).unwrap();

    unsafe {
        let ptr = alloc::alloc(layout);
        *(ptr as *mut usize) = payload_size;
        ptr.add(EXTRA_SIZE) // Pointer to the payload.
    }
}

#[no_mangle]
fn dealloc_implicit(ptr: *mut u8) {
    // `ptr` points at the payload, which is immediately preceded by the size (which does not
    // include the size of the size itself).
    unsafe {
        let base_ptr = ptr.sub(EXTRA_SIZE);
        let payload_size = *(base_ptr as *mut usize);

        let layout = alloc::Layout::from_size_align(payload_size + EXTRA_SIZE, ALIGN).unwrap();
        alloc::dealloc(base_ptr, layout);
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
