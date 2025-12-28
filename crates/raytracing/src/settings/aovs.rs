use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct AOVFlags: u32 {
        const NORMALS = 1 << 0;
        const UV_COORDS = 1 << 1;
        
        const DEBUG = 
            AOVFlags::NORMALS.bits() 
            | AOVFlags::UV_COORDS.bits();
    }
}