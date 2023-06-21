use std::mem;

#[allow(dead_code)]
pub const fn const_assert(ok: bool) {
    let _ = 0 - !ok as usize;
}

#[allow(dead_code)]
pub const fn assert_size<T>(size: usize) {
    let ok = mem::size_of::<T>() == size;
    let _ = 0 - !ok as usize;
}
