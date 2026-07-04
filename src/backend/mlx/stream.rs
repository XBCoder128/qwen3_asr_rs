//! MLX device and stream management.

use super::ffi;
use std::sync::OnceLock;

static DEFAULT_STREAM: OnceLock<MlxStream> = OnceLock::new();

pub struct MlxStream {
    pub(crate) ptr: ffi::mlx_stream,
}

unsafe impl Send for MlxStream {}
unsafe impl Sync for MlxStream {}

impl Drop for MlxStream {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::mlx_stream_free(self.ptr) };
        }
    }
}

pub fn init_mlx(use_gpu: bool) {
    DEFAULT_STREAM.get_or_init(|| {
        let device_type = if use_gpu {
            let mut available = false;
            unsafe { ffi::mlx_metal_is_available(&mut available) };
            if available {
                ffi::mlx_device_type::MLX_GPU
            } else {
                eprintln!("Warning: Metal GPU not available, falling back to CPU");
                ffi::mlx_device_type::MLX_CPU
            }
        } else {
            ffi::mlx_device_type::MLX_CPU
        };

        let device = unsafe { ffi::mlx_device_new_type(device_type, 0) };
        unsafe { ffi::mlx_set_default_device(device) };

        let stream = unsafe { ffi::mlx_stream_new_device(device) };
        unsafe { ffi::mlx_device_free(device) };

        MlxStream { ptr: stream }
    });
}

pub fn default_stream() -> ffi::mlx_stream {
    DEFAULT_STREAM
        .get()
        .expect("MLX not initialized. Call init_mlx() first.")
        .ptr
}

pub fn synchronize() {
    let stream = default_stream();
    unsafe { ffi::mlx_synchronize(stream) };
}

/// Clear the MLX memory cache. MLX retains allocated Metal buffers for reuse,
/// but they are never returned to the OS. Call this after inference to free
/// GPU memory — critical for streaming where inference runs repeatedly.
pub fn clear_cache() {
    unsafe { ffi::mlx_clear_cache() };
}

/// Get current active memory usage in bytes.
pub fn active_memory() -> usize {
    let mut val: usize = 0;
    unsafe { ffi::mlx_get_active_memory(&mut val) };
    val
}

/// Get cache memory usage in bytes.
pub fn cache_memory() -> usize {
    let mut val: usize = 0;
    unsafe { ffi::mlx_get_cache_memory(&mut val) };
    val
}
