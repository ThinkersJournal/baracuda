// SPDX-License-Identifier: MIT OR Apache-2.0
//! CUDA Graph user objects (CUDA 12.0+).
//!
//! Graphs frequently hold references to external resources (allocators,
//! file handles, RAII guards) that must be kept alive for the graph's
//! lifetime. A [`UserObject`] is a refcounted handle + destructor that
//! you can *attach* to a graph via `Graph::retain_user_object`; when
//! the graph releases the last reference, the destructor runs.
//!
//! The Rust safe wrapper owns a `Box<dyn FnOnce() + Send>` trampoline so
//! idiomatic `move`-closures work as destructors.

use core::ffi::c_void;

use baracuda_cuda_sys::{CUuserObject, driver};

use crate::error::{Result, check};

/// One owned reference to a refcounted user object.
///
/// A `UserObject` value owns exactly ONE reference, and the type keeps it that way: [`release`](Self::release) takes `self`, so a
/// reference can be given back once, and [`Drop`] gives back the one this value still owns (none, after `release`). Extra references come
/// from [`duplicate`](Self::duplicate) (another owning value) or from a graph (`Graph::retain_user_object`, whose references belong to the
/// graph, not to this value). The earlier shape (`release(&self, count)` plus a `Drop` that always released once more) let safe code
/// release a reference it did not own, and release the same one twice.
pub struct UserObject {
    /// The handle while this value still owns its reference; `None` once `release` has given it back.
    handle: Option<CUuserObject>,
}

unsafe impl Send for UserObject {}
unsafe impl Sync for UserObject {}

impl core::fmt::Debug for UserObject {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("UserObject")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

unsafe extern "C" fn destroy_trampoline(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    // SAFETY: `user_data` was `Box::into_raw`'d by `UserObject::new`.
    let f: Box<Box<dyn FnOnce() + Send>> =
        unsafe { Box::from_raw(user_data as *mut Box<dyn FnOnce() + Send>) };
    (*f)();
}

impl UserObject {
    /// Create a user object whose destructor is `destroy`; the returned value owns the one initial reference.
    pub fn new<F>(destroy: F) -> Result<Self>
    where
        F: FnOnce() + Send + 'static,
    {
        let boxed: Box<Box<dyn FnOnce() + Send>> = Box::new(Box::new(destroy));
        let raw = Box::into_raw(boxed) as *mut c_void;
        let d = driver()?;
        let cu = d.cu_user_object_create()?;
        let mut object: CUuserObject = core::ptr::null_mut();
        // CUDA currently requires flags == CU_USER_OBJECT_NO_DESTRUCTOR_SYNC (= 1);
        // the destructor may run on any CUDA-internal thread.
        const CU_USER_OBJECT_NO_DESTRUCTOR_SYNC: core::ffi::c_uint = 1;
        let rc = unsafe {
            cu(
                &mut object,
                raw,
                Some(destroy_trampoline),
                1,
                CU_USER_OBJECT_NO_DESTRUCTOR_SYNC,
            )
        };
        if rc != baracuda_cuda_sys::CUresult::SUCCESS {
            // Reclaim the box so we don't leak the closure.
            drop(unsafe { Box::from_raw(raw as *mut Box<dyn FnOnce() + Send>) });
            return Err(crate::error::Error::Status { status: rc });
        }
        Ok(Self {
            handle: Some(object),
        })
    }

    /// Take one more reference and return it as its own owning value (`cuUserObjectRetain` with a count of 1). The destructor runs when
    /// every owning value and every graph reference has been released.
    pub fn duplicate(&self) -> Result<UserObject> {
        let handle = self.owned_handle();
        let d = driver()?;
        let cu = d.cu_user_object_retain()?;
        check(unsafe { cu(handle, 1) })?;
        Ok(UserObject {
            handle: Some(handle),
        })
    }

    /// Give this value's reference back now (`cuUserObjectRelease` with a count of 1); the destructor runs if it was the last one.
    ///
    /// Takes `self`, so the value cannot be used or released again, and `Drop` has nothing left to release. The reference is gone even
    /// when the driver reports an error: there is no handle to retry on.
    ///
    /// Releasing twice does not compile:
    ///
    /// ```compile_fail
    /// # use baracuda_driver::user_object::UserObject;
    /// # fn f(uo: UserObject) {
    /// let _ = uo.release();
    /// let _ = uo.release(); // error[E0382]: use of moved value
    /// # }
    /// ```
    pub fn release(mut self) -> Result<()> {
        let handle = self.owned_handle();
        // Taken before the call: whatever the driver answers, this value no longer owns a reference and `Drop` must not release one.
        self.handle = None;
        let d = driver()?;
        let cu = d.cu_user_object_release()?;
        check(unsafe { cu(handle, 1) })
    }

    /// The handle this value still owns. Only `release` and `Drop` clear it, and both consume or end the value.
    fn owned_handle(&self) -> CUuserObject {
        self.handle
            .expect("a UserObject owns its handle until `release` consumes it")
    }

    /// Raw `CUuserObject` handle. Use with care — owned by `self`.
    #[inline]
    pub fn as_raw(&self) -> CUuserObject {
        self.owned_handle()
    }
}

impl Drop for UserObject {
    fn drop(&mut self) {
        // `take`: a value that was released (or is dropped) gives back its one reference exactly once.
        let Some(handle) = self.handle.take() else {
            return;
        };
        if handle.is_null() {
            return;
        }
        if let Ok(d) = driver()
            && let Ok(cu) = d.cu_user_object_release()
        {
            let _ = unsafe { cu(handle, 1) };
        }
    }
}

// Extend Graph with user-object retention.
impl crate::Graph {
    /// Have this graph take `count` NEW references to `object` (the graph's own: they are not `object`'s reference, which it keeps). When
    /// the graph is destroyed, those references are given back, and the destructor runs if they were the last.
    ///
    /// There is no `flags` argument: the driver's `CU_GRAPH_USER_OBJECT_MOVE` hands the caller's own references to the graph, which a
    /// [`UserObject`] (one owned reference, given back by `Drop`) cannot express without a double release, so the call is always flags 0.
    pub fn retain_user_object(&self, object: &UserObject, count: u32) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_retain_user_object()?;
        check(unsafe { cu(self.as_raw(), object.as_raw(), count, 0) })
    }

    /// Give back `count` references this graph holds on `object` (`cuGraphReleaseUserObject`).
    ///
    /// # Safety
    ///
    /// The graph must hold at least `count` references to `object` taken with [`retain_user_object`](Self::retain_user_object) and not yet
    /// given back: the graph's reference count is the driver's, and giving back more than it holds releases references that belong to
    /// someone else (to a [`UserObject`] value, or to another graph), which is a use of a freed object once the count reaches zero.
    pub unsafe fn release_user_object(&self, object: &UserObject, count: u32) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_release_user_object()?;
        check(unsafe { cu(self.as_raw(), object.as_raw(), count) })
    }
}
