#[cfg(feature = "serde")]
use core::convert::TryFrom;
#[cfg(feature = "serde")]
use core::fmt::Debug;
use core::{cmp, fmt, hash::Hash};
use rand_core::{CryptoRng, OsRng, RngCore};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, sync::Arc, vec::Vec};
#[cfg(feature = "std")]
use std::{boxed::Box, sync::Arc, vec::Vec};

#[cfg(feature = "serde")]
use crate::common::SessionState;
use crate::{
    Counter, CryptoProvider, DRError, DecryptError, Diff, EncryptUninit, Header, KeyPair,
    sync::{DefaultKeyStore, MessageKeyCacheTrait},
};

// TODO: avoid heap allocations in encrypt/decrypt interfaces
// TODO: HeaderEncrypted version

/// The `DoubleRatchet` can encrypt/decrypt messages while providing forward secrecy and
/// post-compromise security.
///
/// The `DoubleRatchet` struct provides an implementation of the Double Ratchet Algorithm as
/// defined in its [specification], including the unspecified symmetric initialization. After
/// initialization (with `new_alice` or `new_bob`) the user can interact with the `DoubleRatchet`
/// using the `ratchet_encrypt` and `ratchet_decrypt` methods, which automatically takes care of
/// deriving the correct keys and updating the internal state.
///
/// # Initialization
///
/// When Alice and Bob want to use the `DoubleRatchet`, they need to initialize it using different
/// constructors. The "Alice" or "Bob" role follows from the design of the authenticated key
/// exchange that is used to initialize the secure communications channel. Two "modes" are
/// possible, depending on whether just one or both of the parties must be able to send the first
/// data message. See `new_alice` and `new_bob` for further details.
///
/// # Provided security
///
/// Conditional on the correct implementation of the `CryptoProvider`, the `DoubleRatchet` provides
/// confidentiality of the plaintext and authentication of both the ciphertext and associated data.
/// It does not provide anonymity, as the headers have to be sent in plain text and are sufficient
/// for identifying the communicating parties. See `CryptoProvider` for further details on the
/// required security properties.
///
/// Forward secrecy (sometimes called the key-erasure property) preserves confidentiality of old
/// messages in case of a device compromise. The `DoubleRatchet` provides forward secrecy by
/// deriving a fresh key for every message: the sender deletes it immediately after encrypting and
/// the receiver deletes it immediately after successful decryption. Messages may arrive out of
/// order, in which case the receiver is able to derive and store the keys for the skipped messages
/// without compromising the forward secrecy of other messages. See [secure deletion] for further
/// discussion.
///
/// Post-compromise security (sometimes called future secrecy or the self-healing property)
/// restores confidentiality of new messages in case of a past device compromise. The
/// `DoubleRatchet` provides future secrecy by generating a fresh `KeyPair` for every reply that is
/// being sent. See [recovery from compromise] for further discussion and [post-compromise] for an
/// in-depth analysis of the subject.
///
/// # Examples
///
/// If Alice is guaranteed to send the first message to Bob, she can initialize her `DoubleRatchet`
/// as shown here, without providing the symmetric `initial_receive` key. It is assumed that
/// `shared_secret` and `bobs_public_key` are the result of some secure key exchange. A higher
/// level protocol may force Alice to always send an empty initial message in order to fully
/// initialize both parties.
///
/// ```
/// # use double_ratchet::{mock, KeyPair, DoubleRatchet, EncryptUninit};
/// # type MyCryptoProvider = mock::CryptoProvider;
/// # let mut csprng = mock::Rng::default();
/// # let bobs_keypair = mock::KeyPair::new(&mut csprng);
/// # let bobs_public_key = bobs_keypair.public().clone();
/// # let shared_secret = [42, 0];
/// type DR = DoubleRatchet<MyCryptoProvider>;
/// /// Alice and Bob have agreed on `shared_secret` and `bobs_public_key`
/// let mut alice = DR::new_alice(&shared_secret, bobs_public_key, None, &mut csprng);
/// let mut bob = DR::new_bob(shared_secret, bobs_keypair, None);
///
/// /// Bob cannot send to Alice
/// assert_eq!(Err(EncryptUninit), bob.try_ratchet_encrypt(b"Hi Alice", b"B2A", &mut csprng));
///
/// /// Alice can send to Bob
/// let (head, ct) = alice.ratchet_encrypt(b"Hello Bob", b"A2B", &mut csprng);
/// let pt = bob.ratchet_decrypt(&head, &ct, b"A2B").unwrap();
/// assert_eq!(&pt[..], b"Hello Bob");
///
/// /// Now Bob can send to Alice
/// let (head, ct) = bob.ratchet_encrypt(b"Hi Alice", b"B2A", &mut csprng);
/// let pt = alice.ratchet_decrypt(&head, &ct, b"B2A").unwrap();
/// assert_eq!(&pt[..], b"Hi Alice");
/// ```
///
/// If it is required that either party can send the first message, the key exchange must provide
/// us with an `extra_shared_secret`.
///
/// ```
/// # use double_ratchet::{mock, KeyPair, DoubleRatchet};
/// # type MyCryptoProvider = mock::CryptoProvider;
/// # let mut csprng = mock::Rng::default();
/// # let bobs_keypair = mock::KeyPair::new(&mut csprng);
/// # let bobs_public_key = bobs_keypair.public().clone();
/// # let shared_secret = [42, 0];
/// # let extra_shared_secret = [42, 0, 0];
/// # type DR = DoubleRatchet<MyCryptoProvider>;
/// let mut alice = DR::new_alice(&shared_secret, bobs_public_key, Some(extra_shared_secret), &mut csprng);
/// let mut bob = DR::new_bob(shared_secret, bobs_keypair, Some(extra_shared_secret));
///
/// /// Either Alice or Bob can send the first message
/// let (head_bob, ct_bob) = bob.ratchet_encrypt(b"Hi Alice", b"from Bob to Alice", &mut csprng);
/// let (head_alice, ct_alice) = alice.ratchet_encrypt(b"Hello Bob", b"from Alice to Bob", &mut csprng);
/// let pt_bob = alice.ratchet_decrypt(&head_bob, &ct_bob, b"from Bob to Alice").unwrap();
/// let pt_alice = bob.ratchet_decrypt(&head_alice, &ct_alice, b"from Alice to Bob").unwrap();
/// assert_eq!(&pt_alice[..], b"Hello Bob");
/// assert_eq!(&pt_bob[..], b"Hi Alice");
/// ```
///
/// [post-compromise]: https://eprint.iacr.org/2016/221
/// [specification]: https://signal.org/docs/specifications/doubleratchet/#double-ratchet-1
/// [secure deletion]: https://signal.org/docs/specifications/doubleratchet/#secure-deletion
/// [recovery from compromise]: https://signal.org/docs/specifications/doubleratchet/#recovery-from-compromise
pub struct DoubleRatchet<CP: CryptoProvider + 'static> {
    id: u64,
    dhs: CP::KeyPair,
    dhr: Option<CP::PublicKey>,
    rk: CP::RootKey,
    cks: Option<CP::ChainKey>,
    ckr: Option<CP::ChainKey>,
    ns: Counter,
    nr: Counter,
    pn: Counter,
    msg_key_cache: Arc<dyn MessageKeyCacheTrait<CP>>,
}

impl<CP> fmt::Debug for DoubleRatchet<CP>
where
    CP: CryptoProvider,
    CP::KeyPair: fmt::Debug,
    CP::PublicKey: fmt::Debug,
    CP::RootKey: fmt::Debug,
    CP::ChainKey: fmt::Debug,
    CP::MessageKey: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "DoubleRatchet {{ id: {:?}, dhs: {:?}, dhr: {:?}, rk: {:?}, cks: {:?}, ckr: {:?}, ns: {:?}, \
             nr: {:?}, pn: {:?}, message_key_cache: {:?} }}",
            self.id,
            self.dhs,
            self.dhr,
            self.rk,
            self.cks,
            self.ckr,
            self.ns,
            self.nr,
            self.pn,
            self.msg_key_cache
        )
    }
}

impl<CP: CryptoProvider> DoubleRatchet<CP> {
    /// Initialize "Alice": the sender of the first message.
    ///
    /// This implements `RatchetInitAlice` as defined in the [specification] when `initial_receive
    /// = None`: after initialization Alice must send a message to Bob before he is able to provide
    /// a reply.
    ///
    /// Alternatively Alice provides an extra symmetric key: `initial_receive = Some(key)`, so that
    /// both Alice and Bob can send the first message. Note however that even when Alice and Bob
    /// initialize this way the initialization is asymmetric in the sense that Alice requires Bob's
    /// public key.
    ///
    /// Either Alice and Bob must supply the same extra symmetric key or both must supply `None`.
    ///
    /// # Security considerations
    ///
    /// For security, initialization through `new_alice` has the following requirements:
    ///  - `shared_secret` must be both *confidential* and *authenticated*
    ///  - `them` must be *authenticated*
    ///  - `initial_receive` is `None` or `Some(key)` where `key` is *confidential* and *authenticated*
    ///
    /// [specification]: https://signal.org/docs/specifications/doubleratchet/#initialization
    pub fn new_alice<R: CryptoRng + RngCore>(
        shared_secret: &CP::RootKey,
        them: CP::PublicKey,
        initial_receive: Option<CP::ChainKey>,
        rng: &mut R,
    ) -> Self {
        let dhs = CP::KeyPair::new(rng);
        let (rk, cks) = CP::kdf_rk(shared_secret, &CP::diffie_hellman(&dhs, &them));
        Self {
            id: rng.next_u64(),
            dhs,
            dhr: Some(them),
            rk,
            cks: Some(cks),
            ckr: initial_receive,
            ns: 0,
            nr: 0,
            pn: 0,
            msg_key_cache: Arc::new(DefaultKeyStore::new()),
        }
    }

    /// Initialize "Bob": the receiver of the first message.
    ///
    /// This implements `RatchetInitBob` as defined in the [specification] when `initial_send =
    /// None`: after initialization Bob must receive a message from Alice before he can send his
    /// first message.
    ///
    /// Alternatively Bob provides an extra symmetric key: `initial_send = Some(key)`, so that both
    /// Alice and Bob can send the first message. Note however that even when Alice and Bob
    /// initialize this way the initialization is asymmetric in the sense that Bob must provide his
    /// public key to Alice.
    ///
    /// Either Alice and Bob must supply the same extra symmetric key or both must supply `None`.
    ///
    /// # Security considerations
    ///
    /// For security, initialization through `new_bob` has the following requirements:
    ///  - `shared_secret` must be both *confidential* and *authenticated*
    ///  - the private key of `us` must remain secret on Bob's device
    ///  - `initial_send` is `None` or `Some(key)` where `key` is *confidential* and *authenticated*
    ///
    /// [specification]: https://signal.org/docs/specifications/doubleratchet/#initialization
    pub fn new_bob(
        shared_secret: CP::RootKey,
        us: CP::KeyPair,
        initial_send: Option<CP::ChainKey>,
    ) -> Self {
        Self {
            id: OsRng.next_u64(),
            dhs: us,
            dhr: None,
            rk: shared_secret,
            cks: initial_send,
            ckr: None,
            ns: 0,
            nr: 0,
            pn: 0,
            msg_key_cache: Arc::new(DefaultKeyStore::new()),
        }
    }

    /// # Returns
    /// the instance `id`
    pub fn id(&self) -> u64 {
        self.id
    }

    /// sets the instance `id`
    pub fn set_id(&mut self, id: u64) {
        self.id = id;
    }

    /// returns a copy of `MessageKeyCacheTrait` instance currently in use
    pub fn message_key_cache(&self) -> Arc<dyn MessageKeyCacheTrait<CP>> {
        self.msg_key_cache.clone()
    }

    /// sets the `MessageKeyCacheTrait` instance for use as part of skipped messages
    /// the can be set immediately after creating the new `DoubleRatchet` instance
    /// or more specifically, before the the instance is used.
    /// if nothing is set, it will default to a memory only KeyCache
    pub fn set_message_key_cache(&mut self, cache: Arc<dyn MessageKeyCacheTrait<CP>>) {
        self.msg_key_cache = cache;
    }

    /// The current public key that can be shared with the other party
    pub fn public_key(&self) -> &CP::PublicKey {
        self.dhs.public()
    }

    /// maximum number of skipped entries to prevent `DoS`
    #[allow(dead_code)]
    pub fn max_skip(&self) -> usize {
        self.msg_key_cache.max_skip()
    }

    /// set the maximum number of skipped entries to prevent `DoS`
    /// this should be called after `set_message_key_cache`
    /// WARNING: this could result in setting the value globally
    /// based on the key cache implementation
    #[allow(dead_code)]
    pub fn set_max_skip(&self, max_skip: usize) {
        self.msg_key_cache.set_max_skip(max_skip);
    }

    /// maximum number of skipped capacity to prevent `DoS`
    #[allow(dead_code)]
    pub fn max_capacity(&self) -> usize {
        self.msg_key_cache.max_capacity()
    }

    /// set the maximum number of skipped entries to prevent `DoS`
    /// this should be called after `set_message_key_cache`
    /// WARNING: this could result in setting the value globally
    /// based on the key cache implementation
    #[allow(dead_code)]
    pub fn set_max_capacity(&self, max_capacity: usize) {
        self.msg_key_cache.set_max_capacity(max_capacity);
    }

    /// allows the export of the current double ratchet state for persistence
    /// WARNING: the includes the current private keys, etc. and should
    /// be used with caution
    #[allow(dead_code)]
    #[cfg(feature = "serde")]
    pub fn session_state(&self) -> SessionState {
        SessionState {
            id: self.id,
            dhs_priv: self.dhs.private_bytes(),
            dhs_pub: self.dhs.public().clone().as_ref().to_vec(),
            dhr: self.dhr.as_ref().map(|v| v.as_ref().to_vec()),
            rk: self.rk.as_ref().to_vec(),
            cks: self.cks.as_ref().map(|v| v.as_ref().to_vec()),
            ckr: self.ckr.as_ref().map(|v| v.as_ref().to_vec()),
            ns: self.ns,
            nr: self.nr,
            pn: self.pn,
        }
    }

    /// Try to encrypt the `plaintext`. See `ratchet_encrypt` for details.
    ///
    /// Fails with `EncryptUninit` when `self` is not yet initialized for encrypting.
    ///
    /// # Errors
    /// `DecryptError`
    pub fn try_ratchet_encrypt<R: CryptoRng + RngCore>(
        &mut self,
        plaintext: &[u8],
        associated_data: &[u8],
        rng: &mut R,
    ) -> Result<(Header<CP::PublicKey>, Vec<u8>), EncryptUninit> {
        if self.can_encrypt() {
            Ok(self.ratchet_encrypt(plaintext, associated_data, rng))
        } else {
            Err(EncryptUninit)
        }
    }

    /// Encrypt the `plaintext`, ratchet forward and return the (header, ciphertext) pair.
    ///
    /// Implements `RatchetEncrypt` as defined in the [specification]. The header should be sent
    /// along the ciphertext in order for the recipient to be able to `ratchet_decrypt`. The
    /// ciphertext is encrypted in some
    /// [AEAD](https://en.wikipedia.org/wiki/Authenticated_encryption) mode, which encrypts the
    /// `plaintext` and authenticates the `plaintext`, `associated_data` and the header.
    ///
    /// The internal state of the `DoubleRatchet` is automatically updated so that the next message
    /// key be sent with a fresh key.
    ///
    /// Note that `rng` is only used for updating the internal state and not for encrypting the
    /// data.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not initialized for sending yet. If this is a concern, use
    /// `try_ratchet_encrypt` instead to avoid panics.
    ///
    /// [specification]: https://signal.org/docs/specifications/doubleratchet/#encrypting-messages
    pub fn ratchet_encrypt<R: CryptoRng + RngCore>(
        &mut self,
        plaintext: &[u8],
        associated_data: &[u8],
        rng: &mut R,
    ) -> (Header<CP::PublicKey>, Vec<u8>) {
        // TODO: is this the correct place for clear_stack_on_return?
        let (h, mk) = self.ratchet_send_chain(rng);
        let mut ad = h.as_ref();
        ad.extend_from_slice(associated_data);
        //let pt = CP::encrypt(&mk, plaintext, &Self::concat(&h, associated_data));
        let pt = CP::encrypt(&mk, plaintext, &ad);
        (h, pt)
    }

    // Are we initialized such that we can encrypt messages?
    fn can_encrypt(&self) -> bool {
        self.cks.is_some() || self.dhr.is_some()
    }

    // Ratcheting forward the DH chain for sending is delayed until the first message in that chain
    // is going to be sent.
    //
    // [specification]: https://signal.org/docs/specifications/doubleratchet/#deferring-new-ratchet-key-generation
    //
    // # Panics
    //
    // Panics if encrypting is not yet initialized
    fn ratchet_send_chain<R: CryptoRng + RngCore>(
        &mut self,
        rng: &mut R,
    ) -> (Header<CP::PublicKey>, CP::MessageKey) {
        if self.cks.is_none() {
            let dhr = self
                .dhr
                .as_ref()
                .expect("not yet initialized for encryption");
            self.dhs = CP::KeyPair::new(rng);
            let (rk, cks) = CP::kdf_rk(&self.rk, &CP::diffie_hellman(&self.dhs, dhr));
            self.rk = rk;
            self.cks = Some(cks);
            self.pn = self.ns;
            self.ns = 0;
        }
        let h = Header {
            dh: self.dhs.public().clone(),
            n: self.ns,
            pn: self.pn,
        };
        let (cks, mk) = CP::kdf_ck(self.cks.as_ref().unwrap());
        self.cks = Some(cks);
        self.ns += 1;
        (h, mk)
    }

    /// Verify-decrypt the `ciphertext`, update `self` and return the plaintext.
    ///
    /// Implements `RatchetDecrypt` as defined in the [specification]. Decryption of the ciphertext
    /// includes verifying the authenticity of the `header`, `ciphertext` and `associated_data`
    /// (optional).
    ///
    /// `self` is automatically updated upon successful decryption. This includes ratcheting
    /// forward the receiving key-chain and DH key-chain (if necessary) and storing the
    /// `MessageKeys` of any skipped messages so these messages can be decrypted if they arrive out
    /// of order.
    ///
    /// Returns a `DecryptError` when the plaintext could not be decrypted: `self` remains
    /// unchanged in that case. There could be many reasons: inspect the returned error-value for
    /// further details.
    ///
    /// [specification]: https://signal.org/docs/specifications/doubleratchet/#decrypting-messages-1
    ///
    /// # Errors
    /// `DecryptError`
    pub fn ratchet_decrypt(
        &mut self,
        header: &Header<CP::PublicKey>,
        ciphertext: &[u8],
        associated_data: &[u8],
    ) -> Result<Vec<u8>, DecryptError> {
        // TODO: is this the correct place for clear_stack_on_return?
        let mut h = header.as_ref();
        h.extend_from_slice(associated_data);
        let (diff, pt) = self.try_decrypt(header, ciphertext, &h)?;
        //self.try_decrypt(header, ciphertext, &Self::concat(&header, associated_data))?;
        self.update(diff, header);
        Ok(pt)
    }

    // The actual decryption. Gets a (non-mutable) reference to self to ensure that the state is
    // not changed. Upon successful decryption the state must be updated. The minimum amount of work
    // is done in order to retrieve the correct `MessageKey`: the returned `Diff` object contains
    // the result of that work to avoid doing the work again.
    fn try_decrypt(
        &self,
        h: &Header<CP::PublicKey>,
        ct: &[u8],
        ad: &[u8],
    ) -> Result<(Diff<CP>, Vec<u8>), DecryptError> {
        use Diff::{CurrentChain, NextChain, OldKey};
        if let Some(mk) = self.msg_key_cache.get(&self.id, &h.dh, h.n) {
            Ok((OldKey, CP::decrypt(&mk, ct, ad)?))
        } else if self.dhr.as_ref() == Some(&h.dh) {
            let (ckr, mut mks) =
                Self::skip_message_keys(self.ckr.as_ref().unwrap(), self.get_current_skip(h)?);
            let mk = mks.pop().unwrap();
            Ok((CurrentChain(ckr, mks), CP::decrypt(&mk, ct, ad)?))
        } else {
            let (rk, ckr) = CP::kdf_rk(&self.rk, &CP::diffie_hellman(&self.dhs, &h.dh));
            let (ckr, mut mks) = Self::skip_message_keys(&ckr, self.get_next_skip(h)?);
            let mk = mks.pop().unwrap();
            Ok((NextChain(rk, ckr, mks), CP::decrypt(&mk, ct, ad)?))
        }
    }

    // Calculate how many messages should be skipped in the current receive chain to get the
    // required `MessageKey`. Also check if `h` is valid.
    fn get_current_skip(&self, h: &Header<CP::PublicKey>) -> Result<usize, DecryptError> {
        let skip =
            h.n.checked_sub(self.nr)
                .ok_or(DecryptError::MessageKeyNotFound)? as usize;
        if self.msg_key_cache.max_skip() < skip {
            Err(DecryptError::SkipTooLarge)
        } else if self.msg_key_cache.can_store(&self.id, &h.dh, skip) {
            Ok(skip)
        } else {
            Err(DecryptError::StorageFull)
        }
    }

    // Calculate how many messages should be skipped in the next receive chain to get the required
    // `MessageKey`. Also check if `h` is valid.
    fn get_next_skip(&self, h: &Header<CP::PublicKey>) -> Result<usize, DecryptError> {
        // without malicious participants this error can only be triggered if the local MessageKey
        // has already been deleted.
        let prev_skip =
            h.pn.checked_sub(self.nr)
                .ok_or(DecryptError::MessageKeyNotFound)? as usize;
        let skip = h.n as usize;
        if self.msg_key_cache.max_skip() < cmp::max(prev_skip, skip) {
            Err(DecryptError::SkipTooLarge)
        } else if self.msg_key_cache.can_store(
            &self.id,
            &h.dh,
            (prev_skip + skip).saturating_sub(1),
        ) {
            Ok(skip)
        } else {
            Err(DecryptError::StorageFull)
        }
    }

    // Update the internal state. Assumes that the validity of `h` has already been checked.
    fn update(&mut self, diff: Diff<CP>, h: &Header<CP::PublicKey>) {
        use Diff::{CurrentChain, NextChain, OldKey};
        match diff {
            OldKey => self.msg_key_cache.remove(&self.id, &h.dh, h.n),
            CurrentChain(ckr, mks) => {
                self.msg_key_cache.extend(self.id, &h.dh, self.nr, mks);
                self.ckr = Some(ckr);
                self.nr = h.n + 1;
            }
            NextChain(rk, ckr, mks) => {
                if self.ckr.is_some() && self.nr < h.pn {
                    let ckr = self.ckr.as_ref().unwrap();
                    let (_, prev_mks) = Self::skip_message_keys(ckr, (h.pn - self.nr - 1) as usize);
                    let dhr = self.dhr.as_ref().unwrap();
                    self.msg_key_cache.extend(self.id, dhr, self.nr, prev_mks);
                }
                self.dhr = Some(h.dh.clone());
                self.rk = rk;
                self.cks = None;
                self.ckr = Some(ckr);
                self.nr = h.n + 1;
                self.msg_key_cache.extend(self.id, &h.dh, 0, mks);
            }
        }
    }

    // Do `skip + 1` ratchet steps in the receive chain. Return the last ChainKey
    // and all computed MessageKeys.
    fn skip_message_keys(ckr: &CP::ChainKey, skip: usize) -> (CP::ChainKey, Vec<CP::MessageKey>) {
        // Note: should use std::iter::unfold (currently still in nightly)
        let mut mks = Vec::with_capacity(skip + 1);
        let (mut ckr, mk) = CP::kdf_ck(ckr);
        mks.push(mk);
        for _ in 0..skip {
            let cm = CP::kdf_ck(&ckr);
            ckr = cm.0;
            mks.push(cm.1);
        }
        (ckr, mks)
    }

    // Concatenate `h` and `ad` in a single byte-vector.
    #[allow(dead_code)]
    fn concat(h: &Header<CP::PublicKey>, ad: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(ad);
        h.extend_bytes_into(&mut v);
        v
    }
}

#[cfg(feature = "serde")]
impl<'a, CP: CryptoProvider> TryFrom<&'a (SessionState, Option<Arc<dyn MessageKeyCacheTrait<CP>>>)>
    for DoubleRatchet<CP>
{
    type Error = DRError;
    fn try_from(
        state: &'a (SessionState, Option<Arc<dyn MessageKeyCacheTrait<CP>>>),
    ) -> Result<Self, Self::Error> {
        let mut instance = Self {
            id: state.0.id,
            dhs: CP::KeyPair::new_from_bytes(&state.0.dhs_priv, &state.0.dhs_pub)?,
            dhr: if let Some(dhr) = state.0.dhr.clone() {
                Some(CP::new_public_key(&dhr).map_err(|_| DRError::InvalidKey)?)
            } else {
                None
            },
            rk: CP::new_root_key(&state.0.rk).map_err(|_| DRError::InvalidKey)?,
            cks: if let Some(cks) = state.0.cks.clone() {
                Some(CP::new_chain_key(&cks).map_err(|_| DRError::InvalidKey)?)
            } else {
                None
            },
            ckr: if let Some(ckr) = state.0.ckr.clone() {
                Some(CP::new_chain_key(&ckr).map_err(|_| DRError::InvalidKey)?)
            } else {
                None
            },
            ns: state.0.ns,
            nr: state.0.nr,
            pn: state.0.pn,
            msg_key_cache: Arc::new(DefaultKeyStore::new()),
        };
        if let Some(key_cache) = state.1.clone() {
            instance.set_message_key_cache(key_cache);
        }
        Ok(instance)
    }
}

// Create a mock CryptoProvider for testing purposes. See `tests/signal.rs` for a proper example
// implementation.
#[cfg(feature = "test")]
#[allow(unused)]
#[allow(missing_docs)]
#[allow(clippy::wildcard_imports)]
pub mod mock {
    use super::*;

    pub type DoubleRatchet = super::DoubleRatchet<CryptoProvider>;
    pub struct CryptoProvider;

    impl super::CryptoProvider for CryptoProvider {
        type KeyPair = KeyPair;
        type PublicKey = PublicKey;
        type SharedSecret = u8;

        type RootKey = [u8; 2];
        type ChainKey = [u8; 3];
        type MessageKey = [u8; 3];

        fn diffie_hellman(us: &KeyPair, them: &PublicKey) -> u8 {
            us.0[0].wrapping_add(them.0[0])
        }

        fn kdf_rk(rk: &[u8; 2], s: &u8) -> ([u8; 2], [u8; 3]) {
            ([rk[0], *s], [rk[0], rk[1], 0])
        }

        fn kdf_ck(ck: &[u8; 3]) -> ([u8; 3], [u8; 3]) {
            ([ck[0], ck[1], ck[2].wrapping_add(1)], *ck)
        }

        fn encrypt(mk: &[u8; 3], pt: &[u8], ad: &[u8]) -> Vec<u8> {
            let mut ct = Vec::from(&mk[..]);
            ct.extend_from_slice(pt);
            ct.extend_from_slice(ad);
            ct
        }

        fn decrypt(
            mk: &[u8; 3],
            ct: &[u8],
            ad: &[u8],
        ) -> Result<Vec<u8>, crate::common::DecryptError> {
            if ct.len() < 3 + ad.len() || ct[..3] != mk[..] || !ct.ends_with(ad) {
                Err(crate::common::DecryptError::DecryptFailure)
            } else {
                Ok(Vec::from(&ct[3..ct.len() - ad.len()]))
            }
        }

        fn new_public_key(key: &[u8]) -> Result<Self::PublicKey, DRError> {
            if key.len() != 1 {
                return Err(DRError::InvalidKey);
            }
            Ok(PublicKey([key[0]]))
        }

        fn new_root_key(key: &[u8]) -> Result<Self::RootKey, DRError> {
            if key.len() != 2 {
                return Err(DRError::InvalidKey);
            }
            Ok([key[0], key[1]])
        }

        fn new_chain_key(key: &[u8]) -> Result<Self::ChainKey, DRError> {
            if key.len() != 3 {
                return Err(DRError::InvalidKey);
            }
            Ok([key[0], key[1], key[2]])
        }
    }

    #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    pub struct PublicKey([u8; 1]);
    impl AsRef<[u8]> for PublicKey {
        fn as_ref(&self) -> &[u8] {
            &self.0
        }
    }

    #[derive(Debug)]
    pub struct KeyPair([u8; 1], PublicKey);
    impl crate::common::KeyPair for KeyPair {
        type PublicKey = PublicKey;
        #[allow(clippy::cast_possible_truncation)]
        fn new<R: rand_core::CryptoRng + rand_core::RngCore>(rng: &mut R) -> Self {
            let n = rng.next_u32() as u8;
            Self([n], PublicKey([n + 1]))
        }

        fn public(&self) -> &PublicKey {
            &self.1
        }

        /// access the private key, this is required for persisting the session state
        #[cfg(feature = "serde")]
        fn private_bytes(&self) -> Vec<u8> {
            self.0.to_vec()
        }

        /// used for reinitialization using the persisted session state
        #[cfg(feature = "serde")]
        fn new_from_bytes(private: &[u8], public: &[u8]) -> Result<Self, DRError>
        where
            Self: Sized,
        {
            if private.len() != 1 || public.len() != 1 {
                return Err(DRError::InvalidData);
            }
            Ok(Self([private[0]], PublicKey([public[0]])))
        }
    }

    // FIXME: this functionality exists already, but breaks the build...
    // use rand::rngs::mock::StepRng;
    #[derive(Default)]
    pub struct Rng(u64);
    impl rand_core::RngCore for Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 += 1;
            self.0
        }
        #[allow(clippy::cast_possible_truncation)]
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }
        fn fill_bytes(&mut self, out: &mut [u8]) {
            rand_core::impls::fill_bytes_via_next(self, out);
        }
        fn try_fill_bytes(&mut self, out: &mut [u8]) -> Result<(), rand_core::Error> {
            self.fill_bytes(out);
            Ok(())
        }
    }
    impl super::CryptoRng for Rng {}
}

#[cfg(test)]
mod tests {
    use core::any::Any;

    use super::*;
    use crate::common::*;

    type DR = DoubleRatchet<mock::CryptoProvider>;

    fn asymmetric_setup(rng: &mut mock::Rng) -> (DR, DR) {
        let secret = [42, 0];
        let pair = mock::KeyPair::new(rng);
        let pubkey = pair.public().clone();
        let alice = DR::new_alice(&secret, pubkey, None, rng);
        let bob = DR::new_bob(secret, pair, None);
        (alice, bob)
    }

    fn symmetric_setup(rng: &mut mock::Rng) -> (DR, DR) {
        let secret = [42, 0];
        let ck_init = [42, 0, 0];
        let pair = mock::KeyPair::new(rng);
        let pubkey = pair.public().clone();
        let alice = DR::new_alice(&secret, pubkey, Some(ck_init), rng);
        let bob = DR::new_bob(secret, pair, Some(ck_init));
        (alice, bob)
    }

    #[test]
    fn test_asymmetric_setup() {
        let mut rng = mock::Rng::default();
        let (mut alice, mut bob) = asymmetric_setup(&mut rng);

        // Alice can encrypt, Bob can't
        let (pt_a, ad_a) = (b"Hi Bobby", b"A2B");
        let (pt_b, ad_b) = (b"What's up Al?", b"B2A");
        let (h_a, ct_a) = alice.ratchet_encrypt(pt_a, ad_a, &mut rng);
        assert_eq!(
            Err(EncryptUninit),
            bob.try_ratchet_encrypt(pt_b, ad_b, &mut rng)
        );
        assert_eq!(
            Ok(Vec::from(&pt_a[..])),
            bob.ratchet_decrypt(&h_a, &ct_a, ad_a)
        );

        // but after decryption Bob can encrypt
        let (h_b, ct_b) = bob.ratchet_encrypt(pt_b, ad_b, &mut rng);
        assert_eq!(
            Ok(Vec::from(&pt_b[..])),
            alice.ratchet_decrypt(&h_b, &ct_b, ad_b)
        );
    }

    #[test]
    fn test_symmetric_setup() {
        let mut rng = mock::Rng::default();
        let (mut alice, mut bob) = symmetric_setup(&mut rng);

        // Alice can encrypt, Bob can't
        let (pt_a, ad_a) = (b"Hi Bobby", b"A2B");
        let (pt_b, ad_b) = (b"What's up Al?", b"B2A");
        let (h_a, ct_a) = alice.ratchet_encrypt(pt_a, ad_a, &mut rng);
        let (h_b, ct_b) = bob.ratchet_encrypt(pt_b, ad_b, &mut rng);
        assert_eq!(
            Ok(Vec::from(&pt_a[..])),
            bob.ratchet_decrypt(&h_a, &ct_a, ad_a)
        );
        assert_eq!(
            Ok(Vec::from(&pt_b[..])),
            alice.ratchet_decrypt(&h_b, &ct_b, ad_b)
        );
    }

    #[test]
    fn test_asymmetric_setup_with_session_state() {
        let mut rng = mock::Rng::default();
        let (mut alice, bob) = asymmetric_setup(&mut rng);
        let bob_session_state = bob.session_state().encode().unwrap();

        // Alice can encrypt, Bob can't
        let (pt_a, ad_a) = (b"Hi Bobby", b"A2B");
        let (pt_b, ad_b) = (b"What's up Al?", b"B2A");
        let (h_a, ct_a) = alice.ratchet_encrypt(pt_a, ad_a, &mut rng);
        let alice_session_state = alice.session_state().encode().unwrap();

        let bob_session = SessionState::decode(&bob_session_state).unwrap();
        let mut bob = DR::try_from(&(bob_session, None)).unwrap();
        assert_eq!(
            Err(EncryptUninit),
            bob.try_ratchet_encrypt(pt_b, ad_b, &mut rng)
        );
        assert_eq!(
            Ok(Vec::from(&pt_a[..])),
            bob.ratchet_decrypt(&h_a, &ct_a, ad_a)
        );

        // but after decryption Bob can encrypt
        let (h_b, ct_b) = bob.ratchet_encrypt(pt_b, ad_b, &mut rng);
        let alice_session = SessionState::decode(&alice_session_state).unwrap();
        let mut alice = DR::try_from(&(alice_session, None)).unwrap();
        assert_eq!(
            Ok(Vec::from(&pt_b[..])),
            alice.ratchet_decrypt(&h_b, &ct_b, ad_b)
        );
    }

    #[test]
    fn symmetric_out_of_order() {
        let mut rng = mock::Rng::default();
        let (mut alice, mut bob) = asymmetric_setup(&mut rng);
        let (ad_a, ad_b) = (b"A2B", b"B2A");

        // Alice's message arrive out of order, some are even missing
        let pt_a_0 = b"Hi Bobby";
        let (h_a_0, ct_a_0) = alice.ratchet_encrypt(pt_a_0, ad_a, &mut rng);
        for _ in 1..9 {
            alice.ratchet_encrypt(b"hello?", ad_a, &mut rng); // drop these messages
        }
        let pt_a_9 = b"are you there?";
        let (h_a_9, ct_a_9) = alice.ratchet_encrypt(pt_a_9, ad_a, &mut rng);
        assert_eq!(
            Ok(Vec::from(&pt_a_9[..])),
            bob.ratchet_decrypt(&h_a_9, &ct_a_9, ad_a)
        );
        assert_eq!(
            Ok(Vec::from(&pt_a_0[..])),
            bob.ratchet_decrypt(&h_a_0, &ct_a_0, ad_a)
        );

        // Bob's replies also arrive out of order
        let pt_b_0 = b"Yes I'm here";
        let (h_b_0, ct_b_0) = bob.ratchet_encrypt(pt_b_0, ad_b, &mut rng);
        for _ in 1..9 {
            bob.ratchet_encrypt(b"why?", ad_b, &mut rng); // drop these messages
        }
        let pt_b_9 = b"Tell me why!!!";
        let (h_b_9, ct_b_9) = bob.ratchet_encrypt(pt_b_9, ad_b, &mut rng);
        assert_eq!(
            Ok(Vec::from(&pt_b_9[..])),
            alice.ratchet_decrypt(&h_b_9, &ct_b_9, ad_b)
        );
        assert_eq!(
            Ok(Vec::from(&pt_b_0[..])),
            alice.ratchet_decrypt(&h_b_0, &ct_b_0, ad_b)
        );
    }

    #[test]
    fn dh_out_of_order() {
        let mut rng = mock::Rng::default();
        let (mut alice, mut bob) = asymmetric_setup(&mut rng);
        let (ad_a, ad_b) = (b"A2B", b"B2A");

        let pt_a_0 = b"Good day Robert";
        let (h_a_0, ct_a_0) = alice.ratchet_encrypt(pt_a_0, ad_a, &mut rng);
        assert_eq!(
            Ok(Vec::from(&pt_a_0[..])),
            bob.ratchet_decrypt(&h_a_0, &ct_a_0, ad_a)
        );
        let pt_a_1 = b"Do you like Rust?";
        let (h_a_1, ct_a_1) = alice.ratchet_encrypt(pt_a_1, ad_a, &mut rng);
        // Bob misses pt_a_1

        let pt_b_0 = b"Salutations Allison";
        let (h_b_0, ct_b_0) = bob.ratchet_encrypt(pt_b_0, ad_b, &mut rng);
        // Alice misses pt_b_0
        let pt_b_1 = b"How is your day going?";
        let (h_b_1, ct_b_1) = bob.ratchet_encrypt(pt_b_1, ad_b, &mut rng);
        assert_eq!(
            Ok(Vec::from(&pt_b_1[..])),
            alice.ratchet_decrypt(&h_b_1, &ct_b_1, ad_b)
        );

        let pt_a_2 = b"My day is fine.";
        let (h_a_2, ct_a_2) = alice.ratchet_encrypt(pt_a_2, ad_a, &mut rng);
        assert_eq!(
            Ok(Vec::from(&pt_a_2[..])),
            bob.ratchet_decrypt(&h_a_2, &ct_a_2, ad_a)
        );
        // now Bob receives pt_a_1
        assert_eq!(
            Ok(Vec::from(&pt_a_1[..])),
            bob.ratchet_decrypt(&h_a_1, &ct_a_1, ad_a)
        );

        let pt_b_2 = b"Yes I like Rust";
        let (h_b_2, ct_b_2) = bob.ratchet_encrypt(pt_b_2, ad_b, &mut rng);
        assert_eq!(
            Ok(Vec::from(&pt_b_2[..])),
            alice.ratchet_decrypt(&h_b_2, &ct_b_2, ad_b)
        );
        // now Alice receives pt_b_0
        assert_eq!(
            Ok(Vec::from(&pt_b_0[..])),
            alice.ratchet_decrypt(&h_b_0, &ct_b_0, ad_b)
        );
    }

    #[test]
    #[should_panic(expected = "not yet initialized for encryption")]
    fn encrypt_error() {
        let mut rng = mock::Rng::default();
        let (_alice, mut bob) = asymmetric_setup(&mut rng);

        assert_eq!(
            Err(EncryptUninit),
            bob.try_ratchet_encrypt(b"", b"", &mut rng)
        );
        bob.ratchet_encrypt(b"", b"", &mut rng);
    }

    #[test]
    fn decrypt_failure() {
        let mut rng = mock::Rng::default();
        let (mut alice, mut bob) = asymmetric_setup(&mut rng);
        let (ad_a, ad_b) = (b"A2B", b"B2A");

        // Next chain
        let (h_a_0, ct_a_0) = alice.ratchet_encrypt(b"Hi Bob", ad_a, &mut rng);
        let mut ct_a_0_err = ct_a_0.clone();
        ct_a_0_err[2] ^= 0x80;
        let mut h_a_0_err = h_a_0.clone();
        h_a_0_err.pn = 1;
        assert_eq!(
            Err(DecryptError::DecryptFailure),
            bob.ratchet_decrypt(&h_a_0, &ct_a_0_err, ad_a)
        );
        assert_eq!(
            Err(DecryptError::DecryptFailure),
            bob.ratchet_decrypt(&h_a_0_err, &ct_a_0, ad_a)
        );
        assert_eq!(
            Err(DecryptError::DecryptFailure),
            bob.ratchet_decrypt(&h_a_0, &ct_a_0, ad_b)
        );

        // Current Chain
        let (h_a_1, ct_a_1) = alice.ratchet_encrypt(b"Hi Bob", ad_a, &mut rng);
        bob.ratchet_decrypt(&h_a_1, &ct_a_1, ad_a).unwrap();
        let (h_a_2, ct_a_2) = alice.ratchet_encrypt(b"Hi Bob", ad_a, &mut rng);
        let mut h_a_2_err = h_a_2.clone();
        h_a_2_err.pn += 1;
        let mut ct_a_2_err = ct_a_2.clone();
        ct_a_2_err[0] ^= 0x04;

        assert_eq!(
            Err(DecryptError::DecryptFailure),
            bob.ratchet_decrypt(&h_a_2, &ct_a_2_err, ad_a)
        );
        assert_eq!(
            Err(DecryptError::DecryptFailure),
            bob.ratchet_decrypt(&h_a_2_err, &ct_a_2, ad_a)
        );
        assert_eq!(
            Err(DecryptError::DecryptFailure),
            bob.ratchet_decrypt(&h_a_2, &ct_a_2, ad_b)
        );

        // Previous chain
        let (h_b, ct_b) = bob.ratchet_encrypt(b"Hi Alice", ad_b, &mut rng);
        alice.ratchet_decrypt(&h_b, &ct_b, ad_b).unwrap();
        let (h_a_3, ct_a_3) = alice.ratchet_encrypt(b"Hi Bob", ad_a, &mut rng);
        bob.ratchet_decrypt(&h_a_3, &ct_a_3, ad_a).unwrap();

        assert_eq!(
            Err(DecryptError::DecryptFailure),
            bob.ratchet_decrypt(&h_a_2, &ct_a_2_err, ad_a)
        );
        assert_eq!(
            Err(DecryptError::DecryptFailure),
            bob.ratchet_decrypt(&h_a_2_err, &ct_a_2, ad_a)
        );
        assert_eq!(
            Err(DecryptError::DecryptFailure),
            bob.ratchet_decrypt(&h_a_2, &ct_a_2, ad_b)
        );
    }

    #[test]
    fn double_sending() {
        // The implementation is unable to consistently detect why decryption fails when receiving
        // double messages: the only requirement should be that *any* error is triggered.

        let mut rng = mock::Rng::default();
        let (mut alice, mut bob) = asymmetric_setup(&mut rng);
        let (ad_a, ad_b) = (b"A2B", b"B2A");

        let (h_a_0, ct_a_0) = alice.ratchet_encrypt(b"Whatever", ad_a, &mut rng);
        bob.ratchet_decrypt(&h_a_0, &ct_a_0, ad_a).unwrap();
        assert!(bob.ratchet_decrypt(&h_a_0, &ct_a_0, ad_a).is_err());

        let (h_b_0, ct_b_0) = bob.ratchet_encrypt(b"Whatever", ad_b, &mut rng);
        alice.ratchet_decrypt(&h_b_0, &ct_b_0, ad_b).unwrap();
        assert!(alice.ratchet_decrypt(&h_b_0, &ct_b_0, ad_b).is_err());
        let (h_a_1, ct_a_1) = alice.ratchet_encrypt(b"Whatever", ad_a, &mut rng);
        bob.ratchet_decrypt(&h_a_1, &ct_a_1, ad_a).unwrap();
        assert!(bob.ratchet_decrypt(&h_a_1, &ct_a_1, ad_a).is_err());
        let (h_b_1, ct_b_1) = bob.ratchet_encrypt(b"Whatever", ad_b, &mut rng);
        alice.ratchet_decrypt(&h_b_1, &ct_b_1, ad_b).unwrap();
        assert!(alice.ratchet_decrypt(&h_b_1, &ct_b_1, ad_b).is_err());

        assert!(bob.ratchet_decrypt(&h_a_0, &ct_a_0, ad_a).is_err());
        assert!(alice.ratchet_decrypt(&h_b_0, &ct_b_0, ad_b).is_err());
    }

    #[test]
    fn invalid_header() {
        let mut rng = mock::Rng::default();
        let (mut alice, mut bob) = asymmetric_setup(&mut rng);
        let (ad_a, ad_b) = (b"A2B", b"B2A");
        let (h_a_0, ct_a_0) = alice.ratchet_encrypt(b"Hi Bob", ad_a, &mut rng);
        bob.ratchet_decrypt(&h_a_0, &ct_a_0, ad_a).unwrap();
        let (h_b_0, ct_b_0) = bob.ratchet_encrypt(b"Hi Alice", ad_b, &mut rng);
        alice.ratchet_decrypt(&h_b_0, &ct_b_0, ad_b).unwrap();
        let (mut h_a_1, ct_a_1) = alice.ratchet_encrypt(b"I will lie to you now", ad_a, &mut rng);
        assert_eq!(h_a_1.pn, 1);
        h_a_1.pn = 0;
        assert!(bob.ratchet_decrypt(&h_a_1, &ct_a_1, ad_a).is_err());
    }

    #[test]
    fn skip_too_large() {
        let mut rng = mock::Rng::default();
        let (mut alice, mut bob) = asymmetric_setup(&mut rng);
        let (ad_a, ad_b) = (b"A2B", b"B2A");
        let (h_a_0, ct_a_0) = alice.ratchet_encrypt(b"Hi Bob", ad_a, &mut rng);
        for _ in 0..=alice.max_skip() {
            alice.ratchet_encrypt(b"Not sending this", ad_a, &mut rng);
        }
        let (h_a_1, ct_a_1) = alice.ratchet_encrypt(b"n > MAXSKIP", ad_a, &mut rng);
        assert_eq!(
            Err(DecryptError::SkipTooLarge),
            bob.ratchet_decrypt(&h_a_1, &ct_a_1, ad_a)
        );
        bob.ratchet_decrypt(&h_a_0, &ct_a_0, ad_a).unwrap();
        let (h_b, ct_b) = bob.ratchet_encrypt(b"Hi Alice", ad_b, &mut rng);
        alice.ratchet_decrypt(&h_b, &ct_b, ad_b).unwrap();
        let (h_a_2, ct_a_2) = alice.ratchet_encrypt(b"pn > MAXSKIP", ad_a, &mut rng);
        assert_eq!(
            Err(DecryptError::SkipTooLarge),
            bob.ratchet_decrypt(&h_a_2, &ct_a_2, ad_a)
        );
    }

    #[test]
    fn storage_full() {
        let mut rng = mock::Rng::default();
        let (mut alice, mut bob) = asymmetric_setup(&mut rng);
        let ad_a = b"A2B";

        let mut stored = 0;
        let mks_capacity = alice.msg_key_cache.max_capacity(); //aka DEFAULT_MKS_CAPACITY
        while stored < mks_capacity {
            for _ in 0..cmp::min(alice.max_skip(), mks_capacity - stored) {
                alice.ratchet_encrypt(b"Not sending this", ad_a, &mut rng);
            }
            let (h_a, ct_a) = alice.ratchet_encrypt(b"Hello Bob", ad_a, &mut rng);
            bob.ratchet_decrypt(&h_a, &ct_a, ad_a).unwrap();
            stored += bob.max_skip();
            // We need to downcast the trait object *inside* the Arc.
            // `&*bob.msg_key_cache` dereferences the Arc to get `&(dyn MessageKeyCacheTrait<...>)`.
            // Since `MessageKeyCacheTrait: Any`, this can be downcast.
            let default_key_store = (&*bob.msg_key_cache as &dyn Any)
                .downcast_ref::<DefaultKeyStore<mock::CryptoProvider>>()
                .unwrap();
            let _ = default_key_store
                .key_cache
                .lock()
                .values()
                .map(|hm| hm.len())
                .sum::<usize>();
        }
        alice.ratchet_encrypt(b"Bob can't store this key anymore", ad_a, &mut rng);
        let (h_a, ct_a) = alice.ratchet_encrypt(b"Gotcha, Bob!", ad_a, &mut rng);
        assert_eq!(
            Err(DecryptError::StorageFull),
            bob.ratchet_decrypt(&h_a, &ct_a, ad_a)
        );
    }

    #[test]
    fn cannot_crash_other() {
        // Malicious parties should not be able to crash the other end (this was an
        // issue in an old implementation).

        let mut rng = mock::Rng::default();
        let (ad_a, ad_b) = (b"A2B", b"B2A");

        let (mut alice, mut bob) = symmetric_setup(&mut rng);
        alice.pn = 10;
        bob.pn = 10;
        let (h_a, ct_a) = alice.ratchet_encrypt(b"not important", ad_a, &mut rng);
        let (h_b, ct_b) = bob.ratchet_encrypt(b"not important", ad_b, &mut rng);
        let _ = alice.ratchet_decrypt(&h_b, &ct_b, ad_b);
        let _ = bob.ratchet_decrypt(&h_a, &ct_a, ad_a);

        let (mut alice, mut bob) = asymmetric_setup(&mut rng);
        alice.pn = 10;
        let (h_a, ct_a) = alice.ratchet_encrypt(b"not important", ad_a, &mut rng);
        let _ = bob.ratchet_decrypt(&h_a, &ct_a, ad_a);
        bob.pn = 10;
        let (h_b, ct_b) = bob.ratchet_encrypt(b"not important", ad_b, &mut rng);
        let _ = alice.ratchet_decrypt(&h_b, &ct_b, ad_b);
    }
}
