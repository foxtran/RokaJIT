use std::ffi::c_char;
use std::ptr::NonNull;

use super::MockEe;
use crate::host::EeHost;

impl EeHost for MockEe {
    fn allocate_memory(&self, size: usize) -> Option<NonNull<u8>> {
        Some(self.fake_alloc(size))
    }

    fn free_memory(&self, _block: NonNull<u8>) {
        // Mock buffers stay owned by `buffers` until the mock drops.
    }

    fn get_int_config_value(&self, _name: &str, default: i32) -> i32 {
        default
    }

    fn get_string_config_value(&self, _name: &str) -> Option<String> {
        None
    }

    fn free_string_config_value(&self, _value: NonNull<c_char>) {}

    fn allocate_slab(&self, size: usize) -> Option<(NonNull<u8>, usize)> {
        Some((self.fake_alloc(size), size))
    }

    fn free_slab(&self, _slab: NonNull<u8>, _actual_size: usize) {}
}
