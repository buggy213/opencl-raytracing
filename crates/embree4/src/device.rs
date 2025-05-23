use std::{ptr::{self, null_mut}, ffi::{c_void, CStr}};

use embree4_sys::{rtcGetDeviceError, rtcNewDevice, rtcReleaseDevice, rtcSetDeviceErrorFunction, RTCDevice, RTCError};

use crate::*;



pub struct Device {
    pub(crate) handle: RTCDevice
}

impl Device {
    extern "C" fn print_errors(
        userPtr: *mut core::ffi::c_void,
        code: RTCError,
        error_str: *const core::ffi::c_char
    ) {
        let error_cstr;
        unsafe {
            error_cstr = CStr::from_ptr(error_str);
        }
        eprintln!("{}", Device::error_from_code(code));
        eprintln!("{:?}", error_cstr);
    }

    pub fn new() -> Device {
        let handle = unsafe { rtcNewDevice(ptr::null()) };
        unsafe {
            rtcSetDeviceErrorFunction(handle, Some(Device::print_errors), null_mut());
        }
        Device { 
            handle
        }
    }

    fn error_from_code(error_code: RTCError) -> &'static str {
        match error_code {
            RTCError_RTC_ERROR_INVALID_OPERATION => "invalid operation",
            RTCError_RTC_ERROR_CANCELLED => "operation canceled",
            RTCError_RTC_ERROR_INVALID_ARGUMENT => "invalid argument",
            RTCError_RTC_ERROR_OUT_OF_MEMORY => "out of memory",
            RTCError_RTC_ERROR_UNSUPPORTED_CPU => "unsupported cpu",
            RTCError_RTC_ERROR_UNKNOWN => "unknown error",
            _ => "unknown error"
        }
    }
      
    pub fn get_error(&self) -> &'static str {
        let error_code = unsafe { rtcGetDeviceError(self.handle) };
        Device::error_from_code(error_code)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        eprintln!("Dropping Embree device");
        unsafe {
            rtcReleaseDevice(self.handle);
        }
    }
}