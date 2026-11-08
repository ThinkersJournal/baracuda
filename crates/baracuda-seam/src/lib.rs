//! Profile v1 kernel-seam handshake, provider side.
//!
//! The `SeamHello` envelope, its validation, negotiation and wire codec live in
//! [`fireline_announce`] (the crate both sides of the seam compile; it merged Fuel's
//! `fuel-kernel-seam-announce` and this crate's former own copy). This crate keeps only what is
//! Baracuda's: the profile it implements ([`BARACUDA_PROFILE`]), the capability word it
//! advertises ([`BARACUDA_CAPABILITIES`]), the envelope built from them ([`baracuda_hello`]) and
//! the provider-side C-ABI entry point [`baracuda_seam_hello`] that Fuel reads. The names below
//! are re-exported so `baracuda_seam::SeamHello` and friends still resolve.
//!
//! The wire did not change when the copy was replaced: the same 56 bytes and the same
//! advertised capability word (the FDX tokens and provision-on-request), pinned byte for byte in
//! `tests::advertised_wire_bytes_are_pinned`, which passed against the old copy first.

pub use fireline_announce::{
    ALLOCATED_CAPABILITIES, Negotiated, PROFILE_V1, Role, SEAM_CAP_CONTRACT_QUERY,
    SEAM_CAP_DLPACK_EXT_AFFINE, SEAM_CAP_DLPACK_EXT_GATHER, SEAM_CAP_DLPACK_EXT_GGML,
    SEAM_CAP_DLPACK_EXT_MX, SEAM_CAP_DLPACK_EXT_SYMBOLIC, SEAM_CAP_DLPACK_EXT_V1,
    SEAM_CAP_EXT_MASK, SEAM_CAP_FEAT_MASK, SEAM_CAP_PROVIDER_ONLY_MASK,
    SEAM_CAP_PROVISION_ON_REQUEST, SEAM_CAP_SUB_MASK, SEAM_ENVELOPE_VERSION, SEAM_MAGIC,
    SEAM_MAX_PROFILES, SeamError, SeamHello, negotiate_with_role,
};

/// The seam profile Baracuda implements (Profile v1).
pub const BARACUDA_PROFILE: u16 = PROFILE_V1;

/// The capabilities Baracuda advertises at first connect: the FDX tokens for its shipped kernel
/// families **plus** [`SEAM_CAP_PROVISION_ON_REQUEST`] (the former `SEAM_CAP_JIT_ON_REQUEST`, bit
/// 32): both halves of the §5 seam are live (FireLine published the `fireline-synth` envelope;
/// Baracuda implements the `Synthesizer` it calls), so a consumer may issue
/// provision-on-request calls. Every bit is one `fireline-announce` has allocated
/// (`tests::advertised_capabilities_are_all_allocated`).
pub const BARACUDA_CAPABILITIES: u64 = SEAM_CAP_DLPACK_EXT_V1
    | SEAM_CAP_DLPACK_EXT_MX
    | SEAM_CAP_DLPACK_EXT_GGML
    | SEAM_CAP_DLPACK_EXT_AFFINE
    | SEAM_CAP_DLPACK_EXT_SYMBOLIC
    | SEAM_CAP_DLPACK_EXT_GATHER
    | SEAM_CAP_PROVISION_ON_REQUEST;

/// Build Baracuda's advertised `SeamHello`.
#[must_use]
pub fn baracuda_hello() -> SeamHello {
    SeamHello::advertise(&[BARACUDA_PROFILE], BARACUDA_CAPABILITIES)
}

/// Profile-v1 provider-side handshake entry point (§3.5). The consumer calls this, then runs the
/// §3.2 negotiation. Fills `*out`; returns `0` on success, `2` if `out` is null (FDX house
/// convention: 2 = invalid argument). Never panics, never allocates.
///
/// # Safety
/// `out` must be a valid, writable pointer to a `SeamHello`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_seam_hello(out: *mut SeamHello) -> i32 {
    if out.is_null() {
        return 2;
    }
    // SAFETY: non-null checked above; caller guarantees writability + alignment.
    unsafe { out.write(baracuda_hello()) };
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The advertised envelope, byte for byte. Pinned BEFORE the move onto `fireline-announce`
    /// (against this crate's own former `SeamHello`) and kept identical after it: the wire did not
    /// change.
    #[test]
    fn advertised_wire_bytes_are_pinned() {
        let mut want = [0u8; 56];
        want[0..4].copy_from_slice(&[0x53, 0x45, 0x41, 0x4D]); // "SEAM"
        want[4] = 1; // envelope_version
        want[8] = 1; // profiles_len
        want[10] = 1; // profiles[0] = 1
        want[48..56].copy_from_slice(&0x0000_0001_0000_003Fu64.to_le_bytes()); // bits 0-5 + bit 32
        assert_eq!(baracuda_hello().to_wire_bytes(), want);
        assert_eq!(BARACUDA_CAPABILITIES, 0x0000_0001_0000_003F);
    }

    #[test]
    fn advertised_capabilities_are_all_allocated() {
        // A provisional or unallocated bit must not be advertised by accident.
        assert_eq!(baracuda_hello().unallocated_capabilities(), 0);
    }

    #[test]
    fn hello_advertises_profile_v1() {
        let h = baracuda_hello();
        assert_eq!(h.validate(), Ok(&[BARACUDA_PROFILE][..]));
        assert_eq!(h.magic, SEAM_MAGIC);
        assert_eq!(h.envelope_version, 1);
        assert_eq!(h.profiles_len, 1);
        assert_eq!(h.profiles[0], 1);
        assert_eq!(h.profiles[1], 0); // unused entries zeroed
        assert_eq!(h.reserved, [0; 3]);
        assert_eq!(h.reserved1, [0; 6]);
        assert_ne!(h.capabilities & SEAM_CAP_PROVISION_ON_REQUEST, 0);
        assert_ne!(h.capabilities & SEAM_CAP_DLPACK_EXT_GGML, 0);
    }

    #[test]
    fn c_abi_fills_out_param() {
        let mut hello = SeamHello::advertise(&[], 0);
        let rc = unsafe { baracuda_seam_hello(&mut hello) };
        assert_eq!(rc, 0);
        assert_eq!(hello, baracuda_hello());
        // null: typed error, no panic.
        assert_eq!(unsafe { baracuda_seam_hello(core::ptr::null_mut()) }, 2);
    }

    #[test]
    fn envelope_is_56_bytes() {
        assert_eq!(core::mem::size_of::<SeamHello>(), 56);
        assert_eq!(core::mem::align_of::<SeamHello>(), 8);
    }

    /// The real counterparty: a consumer built on `fireline-announce` (what Fuel compiles)
    /// negotiates with Baracuda's hello as it arrives over the wire, not only through this crate's
    /// own parser.
    #[test]
    fn a_fireline_consumer_negotiates_with_the_baracuda_provider() {
        // A consumer writes the provider-only (FEAT) bits as zero.
        let consumer = SeamHello::profile_v1(SEAM_CAP_DLPACK_EXT_V1 | SEAM_CAP_DLPACK_EXT_GGML);
        let remote = SeamHello::from_wire_bytes(&baracuda_hello().to_wire_bytes())
            .expect("baracuda's hello parses");
        let n = negotiate_with_role(&consumer, &remote, Role::Consumer).expect("negotiates");
        assert_eq!(n.profile, 1);
        // EXT bits are live only when both sides set them; the provider-only bit is the provider's.
        assert_eq!(
            n.capabilities,
            SEAM_CAP_DLPACK_EXT_V1 | SEAM_CAP_DLPACK_EXT_GGML | SEAM_CAP_PROVISION_ON_REQUEST
        );
        // From the provider's own seat the same exchange yields the same live word.
        let theirs = SeamHello::from_wire_bytes(&consumer.to_wire_bytes()).unwrap();
        let m = negotiate_with_role(&baracuda_hello(), &theirs, Role::Provider).unwrap();
        assert_eq!(m, n);
    }

    #[test]
    fn a_consumer_with_no_common_profile_hard_fails() {
        let consumer = SeamHello::advertise(&[2], 0);
        let err = negotiate_with_role(&consumer, &baracuda_hello(), Role::Consumer).unwrap_err();
        assert!(matches!(err, SeamError::VersionMismatch { .. }), "{err:?}");
    }

    #[test]
    fn a_damaged_hello_is_rejected_on_the_wire() {
        let mut bytes = baracuda_hello().to_wire_bytes();
        bytes[0] ^= 0xFF;
        assert!(matches!(
            SeamHello::from_wire_bytes(&bytes),
            Err(SeamError::BadMagic(_))
        ));
        assert_eq!(
            SeamHello::from_wire_bytes(&bytes[..55]),
            Err(SeamError::Truncated)
        );
    }
}
