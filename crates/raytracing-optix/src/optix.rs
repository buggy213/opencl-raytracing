#[repr(C)]
struct OptixDeviceContextInner {
    _data: (),
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>
}

#[repr(C)]
pub(crate) struct OptixDeviceContext(*mut OptixDeviceContextInner);

#[repr(C)]
pub(crate) struct OptixTraversableHandle(usize);