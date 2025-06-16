//! The `SignalDoubleRatchet` provides an example of a secure implementation of the
//! `double_ratchet::CryptoProvider`. This example code is based on the recommended cryptographic
//! algorithms of the [specification]. I have not checked if the code is fully compatible with that
//! of the [Signal application](https://signal.org/) but I suspect that it is.
//!
//! For the public key cryptography part of the `CryptoProvider` I used the
//! [x25519-dalek](https://docs.rs/x25519-dalek/) crate. The implementation boils down to thin
//! wrappers around the provided types and methods. For the symmetric part I created
//! `SymmetricKey`: a newtype wrapper around a 32-byte array. A fully secure implementation of the
//! `DoubleRatchet` may take extra steps to provide security, including but not limited to the
//! examples I have implemented here:
//!  - Prevent memory content leakages using [`clear_on_drop`].
//!  - A custom implementation of [`Debug`] so secret bytes are never written to error logs when
//!    compiled in release mode.
//! I am no expert, so if this is insufficient or can otherwise be improved please let me know. For
//! example, I suspect that the `std::pin::Pin` method can be helpful here.
//!
//! Note that the `MessageKey` for Signal is just 32 bytes which acts as input to another KDF which
//! computes an encryption key, mac key and initialization vector. This last key derivation is done
//! in the `CryptoProvider::encrypt` and `CryptoProvider::decrypt` functions. The advantage of this
//! solution over having a `MessageKey` consisting of a 3-tuple of keys is that this improves the
//! speed of symmetric ratcheting in case a message arrives out of order (or a malicious message
//! with a high skip value arrives as part of a denial-of-service attempt).
//!
//! [`clear_on_drop`]: https://crates.io/crates/clear_on_drop
//! [specification]: https://signal.org/docs/specifications/doubleratchet/#recommended-cryptographic-algorithms

use aes::Aes256;
use async_trait::async_trait;
use cbc::cipher::block_padding::Pkcs7;
use cipher::{BlockDecryptMut, BlockEncryptMut, KeyIvInit};
use cipher::generic_array::{typenum::U32, GenericArray};
use clear_on_drop::clear::Clear;
use double_ratchet::async_::{self as dr, KeyPair as _};
use double_ratchet::DecryptError;
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use rand_core::{CryptoRng, RngCore, OsRng};
use sha2::Sha256;
use std::convert::TryInto;
use std::fmt;
use std::hash::{Hash, Hasher};
use subtle::ConstantTimeEq;
use x25519_dalek::{self, SharedSecret};

pub type SignalDR = dr::DoubleRatchet<SignalCryptoProvider>;

type Aes256CbcEnc = cbc::Encryptor<Aes256>;
type Aes256CbcDec = cbc::Decryptor<Aes256>;

pub struct SignalCryptoProvider;

#[async_trait]
impl dr::CryptoProvider for SignalCryptoProvider {
    type PublicKey = PublicKey;
    type KeyPair = KeyPair;
    type SharedSecret = SharedSecret;

    type RootKey = SymmetricKey;
    type ChainKey = SymmetricKey;
    type MessageKey = SymmetricKey;

    fn diffie_hellman(us: &KeyPair, them: &PublicKey) -> SharedSecret {
        us.private.diffie_hellman(&them.0)
    }

    fn kdf_rk(rk: &SymmetricKey, s: &SharedSecret) -> (SymmetricKey, SymmetricKey) {
        let salt = Some(rk.0.as_slice());
        let ikm = s.as_bytes();
        let prk = Hkdf::<Sha256>::new(salt, ikm);
        let info = &b"WhisperRatchet"[..];
        let mut okm = [0; 64];
        prk.expand(&info, &mut okm).unwrap();
        let rk = GenericArray::<u8, U32>::clone_from_slice(&okm[..32]);
        let ck = GenericArray::<u8, U32>::clone_from_slice(&okm[32..]);
        (SymmetricKey(rk), SymmetricKey(ck))
    }

    fn kdf_ck(ck: &SymmetricKey) -> (SymmetricKey, SymmetricKey) {
        let key = ck.0.as_slice();
        let mut mac = Hmac::<Sha256>::new_from_slice(key).unwrap();
        mac.update(&[0x01]);
        let mk = mac.finalize().into_bytes();
        let mut mac = Hmac::<Sha256>::new_from_slice(key).unwrap();
        mac.update(&[0x02]);
        let ck = mac.finalize().into_bytes();
        (SymmetricKey(ck), SymmetricKey(mk))
    }

    async fn encrypt(key: &SymmetricKey, pt: &[u8], ad: &[u8]) -> Vec<u8> {
        let ikm = key.0.as_slice();
        let prk = Hkdf::<Sha256>::new(None, ikm);
        let info = b"WhisperMessageKeys";
        let mut okm = [0; 80];
        prk.expand(info, &mut okm).unwrap();
        let ek = GenericArray::clone_from_slice(&okm[..32]);
        let mk: GenericArray<u8, U32> = GenericArray::clone_from_slice(&okm[32..64]);
        let iv = GenericArray::clone_from_slice(&okm[64..]);
        let cipher = Aes256CbcEnc::new(&ek, &iv);
        let mut ct = cipher.encrypt_padded_vec_mut::<Pkcs7>(pt);
        let mut mac = Hmac::<Sha256>::new_from_slice(&mk).unwrap();
        mac.update(ad);
        mac.update(&ct);
        let tag = mac.finalize().into_bytes();
        ct.extend((&tag[..8]).into_iter());

        okm.clear();
        ct
    }

    async fn decrypt(key: &SymmetricKey, ct: &[u8], ad: &[u8]) -> Result<Vec<u8>, DecryptError> {
        let ikm = key.0.as_slice();
        let prk = Hkdf::<Sha256>::new(None, ikm);
        let info = b"WhisperMessageKeys";
        let mut okm = [0; 80];
        prk.expand(info, &mut okm).unwrap();
        let dk = GenericArray::clone_from_slice(&okm[..32]);
        let mk: GenericArray<u8, U32> = GenericArray::clone_from_slice(&okm[32..64]);
        let iv = GenericArray::clone_from_slice(&okm[64..]);

        let ct_len = ct.len() - 8;
        let mut mac = Hmac::<Sha256>::new_from_slice(&mk).unwrap();
        mac.update(ad);
        mac.update(&ct[..ct_len]);
        let tag = mac.finalize().into_bytes();
        if bool::from(!(&tag[..8]).ct_eq(&ct[ct_len..])) {
            okm.clear();
            return Err(DecryptError::DecryptFailure);
        }
        let cipher = Aes256CbcDec::new(&dk, &iv);
        if let Ok(pt) = cipher.decrypt_padded_vec_mut::<Pkcs7>(&ct[..ct_len]) {
            okm.clear();
            Ok(pt)
        } else {
            okm.clear();
            Err(DecryptError::DecryptFailure)
        }
    }
    
    fn new_public_key(key: &[u8]) -> Result<Self::PublicKey, double_ratchet::DRError> {
        let key: [u8; 32] = key.try_into().map_err(|_| double_ratchet::DRError::InvalidKey)?;
        Ok(PublicKey(x25519_dalek::PublicKey::from(key)))
    }
    
    fn new_root_key(key: &[u8]) -> Result<Self::RootKey, double_ratchet::DRError> {
        let key= GenericArray::<u8, U32>::clone_from_slice(key);
        Ok(SymmetricKey(key))
    }
    
    fn new_chain_key(key: &[u8]) -> Result<Self::ChainKey, double_ratchet::DRError> {
        let key= GenericArray::<u8, U32>::clone_from_slice(key);
        Ok(SymmetricKey(key))
    }
}

#[derive(Clone, Debug)]
pub struct PublicKey(x25519_dalek::PublicKey);

impl Eq for PublicKey {}

impl PartialEq for PublicKey {
    fn eq(&self, other: &PublicKey) -> bool {
        self.0.as_bytes() == other.0.as_bytes()
    }
}

impl Hash for PublicKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_bytes().hash(state);
    }
}

impl<'a> From<&'a x25519_dalek::StaticSecret> for PublicKey {
    fn from(private: &'a x25519_dalek::StaticSecret) -> PublicKey {
        PublicKey(x25519_dalek::PublicKey::from(private))
    }
}

impl AsRef<[u8]> for PublicKey {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

pub struct KeyPair {
    private: x25519_dalek::StaticSecret,
    public: PublicKey,
}

impl fmt::Debug for KeyPair {
    #[cfg(debug_assertions)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "KeyPair {{ private (bytes): {:?}, public: {:?} }}",
            self.private.to_bytes(),
            self.public
        )
    }

    #[cfg(not(debug_assertions))]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "KeyPair {{ private (bytes): <hidden bytes>, public: {:?} }}",
            self.public
        )
    }
}

impl dr::KeyPair for KeyPair {
    type PublicKey = PublicKey;

    fn new<R: CryptoRng + RngCore>(rng: &mut R) -> KeyPair {
        let private = x25519_dalek::StaticSecret::random_from_rng(rng);
        let public = PublicKey::from(&private);
        KeyPair { private, public }
    }

    fn public(&self) -> &PublicKey {
        &self.public
    }
    
    fn private_bytes(&self) -> Vec<u8> {
        self.private.to_bytes().to_vec()
    }
    
    fn new_from_bytes(private: &[u8], public: &[u8]) -> Result<Self, double_ratchet::DRError>
    where
        Self: Sized 
    {
        let private: [u8; 32] = private.try_into().map_err(|_| double_ratchet::DRError::InvalidKey)?;
        let public: [u8; 32] = public.try_into().map_err(|_| double_ratchet::DRError::InvalidKey)?;
        Ok(KeyPair {
            private: x25519_dalek::StaticSecret::from(private),
            public: PublicKey::from(&x25519_dalek::StaticSecret::from(public)),
        })
    }
}

#[derive(Default, Clone, Hash)]
pub struct SymmetricKey(GenericArray<u8, U32>);

impl fmt::Debug for SymmetricKey {
    #[cfg(debug_assertions)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SymmetricKey({:?})", self.0)
    }

    #[cfg(not(debug_assertions))]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SymmetricKey(<hidden bytes>)")
    }
}

impl AsRef<[u8]> for SymmetricKey {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}


impl Drop for SymmetricKey {
    fn drop(&mut self) {
        self.0.clear();
    }
}

#[tokio::test]
async fn signal_session() {
    let mut rng = OsRng;
    let (ad_a, ad_b) = (b"A2B:SessionID=42", b"B2A:SessionID=42");

    // Copy some values (these are usually the outcome of an X3DH key exchange)
    let bobs_prekey = KeyPair::new(&mut rng);
    let bobs_public_prekey = bobs_prekey.public().clone();
    let shared = SymmetricKey(GenericArray::<u8, U32>::clone_from_slice(
        b"Output of a X3DH key exchange...",
    ));

    // Alice fetches Bob's prekey bundle and completes her side of the X3DH handshake
    let mut alice = SignalDR::new_alice(&shared, bobs_public_prekey, None, &mut rng);
    // Alice creates her first message to Bob
    let pt_a_0 = b"Hello Bob";
    let (h_a_0, ct_a_0) = alice.ratchet_encrypt(pt_a_0, ad_a, &mut rng).await;
    // Alice creates an initial message containing `h_a_0`, `ct_a_0` and other X3DH information

    // Bob receives the message and finishes his side of the X3DH handshake
    let mut bob = SignalDR::new_bob(shared, bobs_prekey, None);
    // Bob can now decrypt the initial message
    assert_eq!(
        Ok(Vec::from(&b"Hello Bob"[..])),
        bob.ratchet_decrypt(&h_a_0, &ct_a_0, ad_a).await
    );
    // Bob is now fully initialized: both sides can send and receive message

    let pt_a_1 = b"I will send this later";
    let (h_a_1, ct_a_1) = alice.ratchet_encrypt(pt_a_1, ad_a, &mut rng).await;
    let pt_b_0 = b"My first reply";
    let (h_b_0, ct_b_0) = bob.ratchet_encrypt(pt_b_0, ad_b, &mut rng).await;
    assert_eq!(
        Ok(Vec::from(&pt_b_0[..])),
        alice.ratchet_decrypt(&h_b_0, &ct_b_0, ad_b).await
    );
    let pt_a_2 = b"What a boring conversation";
    let (h_a_2, _ct_a_2) = alice.ratchet_encrypt(pt_a_2, ad_a, &mut rng).await;
    let pt_a_3 = b"Don't you agree?";
    let (h_a_3, ct_a_3) = alice.ratchet_encrypt(pt_a_3, ad_a, &mut rng).await;
    assert_eq!(
        Ok(Vec::from(&pt_a_3[..])),
        bob.ratchet_decrypt(&h_a_3, &ct_a_3, ad_a).await
    );

    let pt_b_1 = b"Agree with what?";
    let (h_b_1, ct_b_1) = bob.ratchet_encrypt(pt_b_1, ad_b, &mut rng).await;
    assert_eq!(
        Ok(Vec::from(&pt_b_1[..])),
        alice.ratchet_decrypt(&h_b_1, &ct_b_1, ad_b).await
    );

    assert_eq!(
        Ok(Vec::from(&pt_a_1[..])),
        bob.ratchet_decrypt(&h_a_1, &ct_a_1, ad_a).await
    );

    // No resending (that key is already deleted)
    assert!(bob.ratchet_decrypt(&h_a_1, &ct_a_1, ad_a).await.is_err());
    // No fake messages
    assert!(bob
        .ratchet_decrypt(&h_a_2, b"Incorrect ciphertext", ad_a).await
        .is_err());
}

