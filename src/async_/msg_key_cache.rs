use async_trait::async_trait;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use core::{
    any::Any,
    cell::UnsafeCell,
    fmt::{self, Debug},
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicBool, Ordering},
};
use hashbrown::HashMap;

#[cfg(feature = "std")]
use std::{boxed::Box, vec::Vec};

use crate::common::{Counter, CryptoProvider, DEFAULT_MAX_SKIP, DEFAULT_MKS_CAPACITY};

/// A `MessageKeyCacheTrait` holds the skipped `MessageKey`s.
///
/// When messages can arrive out of order, the `DoubleRatchet` must store the `MessageKeys`
/// corresponding to the messages that were skipped over. See also the [specification] for further
/// discussion.
///
/// [specification]: https://signal.org/docs/specifications/doubleratchet/#deletion-of-skipped-message-keys
#[async_trait]
pub trait MessageKeyCacheTrait<CP: CryptoProvider>: Any + Debug + Send + Sync {
    /// maximum number of skipped entries between an alice & bob for a specific chain to prevent `DoS`
    fn max_skip(&self) -> usize;
    /// set the maximum entries for skip
    fn set_max_skip(&self, max_skip: usize);
    /// maximum number number of entries in total between alice & bob
    fn max_capacity(&self) -> usize;
    /// set the maximum entries for total capacity
    fn set_max_capacity(&self, max_capacity: usize);

    /// Get the `MessageKey` at `(dh, n)` if it is stored
    async fn get(&self, id: &u64, dh: &CP::PublicKey, n: Counter) -> Option<CP::MessageKey>;

    /// Do `n` more `MessageKeys` fit in the `KeyStore` for `dh` chain?
    async fn can_store(&self, id: &u64, dh: &CP::PublicKey, n: usize) -> bool;

    /// Extend the storage with `mks`
    ///
    /// Keys are stored at `dh` and `n` counting upwards:
    ///   (dh, n  ): mks[0]
    ///   (dh, n+1): mks[1]
    ///   ...
    async fn extend(&self, id: u64, dh: &CP::PublicKey, n: Counter, mks: Vec<CP::MessageKey>);

    /// Remove the `MessageKey` at index `(dh, n)`
    ///
    /// Assumes the `MessageKey` is indeed stored.
    async fn remove(&self, id: &u64, dh: &CP::PublicKey, n: Counter);
}

///
/// Default implementation for `MessageKeyCacheTrait`
///
pub struct DefaultKeyStore<CP: CryptoProvider> {
    /// cache stores the skipped keys
    /// `[id : [PublicKey : [ Counter : MessageKey ] ]`
    #[allow(clippy::type_complexity)]
    pub key_cache:
        SpinMutex<HashMap<u64, HashMap<CP::PublicKey, HashMap<Counter, CP::MessageKey>>>>,
    /// see DEFAULT_MAX_SKIP
    pub max_skip: SpinMutex<usize>,
    /// see DEFAULT_MKS_CAPACITY
    pub message_key_max_capacity: SpinMutex<usize>,
}

impl<CP> fmt::Debug for DefaultKeyStore<CP>
where
    CP: CryptoProvider,
    CP::PublicKey: fmt::Debug,
    CP::MessageKey: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "KeyStore Max skip {}, Max capacity {},\nkey_cache({:?})",
            *self.max_skip.lock(),
            *self.message_key_max_capacity.lock(),
            self.key_cache.lock().raw_entry()
        )
    }
}

impl<CP: CryptoProvider + 'static> DefaultKeyStore<CP> {
    /// new instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            key_cache: SpinMutex::new(HashMap::new()),
            max_skip: SpinMutex::new(DEFAULT_MAX_SKIP),
            message_key_max_capacity: SpinMutex::new(DEFAULT_MKS_CAPACITY),
        }
    }
}

impl<CP: CryptoProvider + 'static> Default for DefaultKeyStore<CP> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<CP: CryptoProvider + 'static> MessageKeyCacheTrait<CP> for DefaultKeyStore<CP> {
    fn max_skip(&self) -> usize {
        *self.max_skip.lock()
    }

    fn max_capacity(&self) -> usize {
        *self.message_key_max_capacity.lock()
    }

    #[allow(dead_code)]
    fn set_max_skip(&self, max_skip: usize) {
        let mut ms = self.max_skip.lock();
        *ms = max_skip;
    }

    #[allow(dead_code)]
    fn set_max_capacity(&self, max_capacity: usize) {
        let mut mk = self.message_key_max_capacity.lock();
        *mk = max_capacity;
    }

    async fn get(&self, id: &u64, dh: &CP::PublicKey, n: Counter) -> Option<CP::MessageKey> {
        self.key_cache
            .lock()
            .get(id)
            .and_then(|hm| hm.get(dh)?.get(&n).cloned())
    }

    async fn can_store(&self, id: &u64, _dh: &CP::PublicKey, n: usize) -> bool {
        let current = self
            .key_cache
            .lock()
            .get(id)
            .map_or(0, |hm| hm.values().map(HashMap::len).sum());
        current + n <= self.max_capacity()
    }

    async fn extend(&self, id: u64, dh: &CP::PublicKey, n: Counter, mks: Vec<CP::MessageKey>) {
        let values = (n..).zip(mks);
        let mut key_cache = self.key_cache.lock();
        let reference = key_cache.entry(id).or_default();
        if let Some(v) = reference.get_mut(dh) {
            v.extend(values);
        } else {
            reference.insert(dh.clone(), values.collect());
        }
    }

    async fn remove(&self, id: &u64, dh: &CP::PublicKey, n: Counter) {
        let mut key_cache = self.key_cache.lock();
        debug_assert!(key_cache.get(id).is_some_and(|hm| hm.contains_key(dh)));
        if let Some(h1) = key_cache.get_mut(id) {
            if let Some(h2) = h1.get_mut(dh) {
                if h2.len() == 1 {
                    _ = h1.remove(dh);
                } else {
                    _ = h2.remove(&n);
                }
            }
        }
    }
}

impl<CP: CryptoProvider + 'static> Into<Box<dyn MessageKeyCacheTrait<CP>>> for DefaultKeyStore<CP> {
    fn into(self) -> Box<dyn MessageKeyCacheTrait<CP>> {
        Box::new(self)
    }
}

/// Simplified `Mutex` impl that does not depend on `std`
/// Warning: this should not be used for production purposes
pub struct SpinMutex<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

/// Simplified `MutexGuard` impl that does not depend on `std`
/// Warning: this should not be used for production purposes
pub struct SpinMutexGuard<'a, T> {
    mutex: &'a SpinMutex<T>,
}

impl<T> SpinMutex<T> {
    /// Simplified `Mutex` impl that does not depend on `std`
    /// Warning: this should not be used for production purposes
    pub const fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    /// Simplified `Mutex` impl that does not depend on `std`
    /// Warning: this should not be used for production purposes
    pub fn lock(&self) -> SpinMutexGuard<T> {
        // Spin until we can acquire the lock
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // Hint to the CPU that we're spinning
            core::hint::spin_loop();
        }

        SpinMutexGuard { mutex: self }
    }
}

impl<T> Deref for SpinMutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mutex.data.get() }
    }
}

impl<T> DerefMut for SpinMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<T> Drop for SpinMutexGuard<'_, T> {
    fn drop(&mut self) {
        self.mutex.locked.store(false, Ordering::Release);
    }
}

unsafe impl<T: Send> Send for SpinMutex<T> {}
unsafe impl<T: Send> Sync for SpinMutex<T> {}
