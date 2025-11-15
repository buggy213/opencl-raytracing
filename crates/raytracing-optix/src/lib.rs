unsafe extern "C" {
    unsafe fn test();
}

pub fn test_rs() {
    unsafe { test(); }
}