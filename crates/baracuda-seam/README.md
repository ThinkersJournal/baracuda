# baracuda-seam

Profile v1 kernel-seam handshake.

Baracuda's side of the handshake: the `SeamHello` it advertises (the envelope, validation, negotiation and wire codec come from `fireline-announce` `~0.1` and are re-exported here) and the `baracuda_seam_hello()` C entry point. Wire unchanged. See the crate documentation for the capability bits.

Part of [baracuda](https://github.com/ciresnave/baracuda); licensed under MIT OR Apache-2.0 (see `LICENSE-MIT`, `LICENSE-APACHE`).
