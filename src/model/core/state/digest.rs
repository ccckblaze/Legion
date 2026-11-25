//! Digests of one slot's recurrent state, for exactness checks on a GPU (`legion serve
//! --prefix-state-digests`, #46): the bits of a state saved inside a prompt step against one
//! saved at a step's end, without dumping hundreds of MB.

use super::StatePool;

/// A 64-bit FNV-1a digest of `bytes`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

impl StatePool {
    /// The digest ([`fnv1a`]) of each of `slot`'s state tensors, named by the tensor (per
    /// recurrent layer its conv history `conv_state-<il>` and delta-rule state
    /// `ssm_state-<il>`, then the per-layer embedding's history), and of its token history
    /// (`tokens`). Reads the slot back (from the device): the caller orders this after the work
    /// that writes it.
    pub fn slot_digests(&self, slot: usize) -> Vec<(String, u64)> {
        assert!(slot < self.n_slots, "slot {slot} out of range");
        let mut out: Vec<(String, u64)> = self
            .tensors()
            .map(|t| {
                // Slots in dim 3, each contiguous.
                let per = t.nb()[3];
                (t.name(), fnv1a(&t.bytes_at(slot * per, per)))
            })
            .collect();
        let tokens = self.peek_tokens(slot);
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        out.push(("tokens".to_string(), fnv1a(&bytes)));
        out
    }

    /// Bytes one [`Self::append_slot_bytes`] snapshot takes: every state tensor's
    /// slot and the token history window.
    pub fn slot_region_bytes(&self) -> usize {
        self.tensors().map(|t| t.nb()[3]).sum::<usize>() + self.n_hist * 4
    }

    /// Appends `slot`'s state tensor bytes, in [`Self::slot_digests`]'s tensor order
    /// then its token history window: the conversation cold tier's state snapshot
    /// shape ([`Self::restore_slot_bytes`] is its inverse). Reads each tensor back (from
    /// the device) on the calling thread; the caller orders this after the copies that
    /// wrote the slot.
    pub fn append_slot_bytes(&self, slot: usize, out: &mut Vec<u8>) {
        assert!(slot < self.n_slots, "slot {slot} out of range");
        for t in self.tensors() {
            let per = t.nb()[3];
            out.extend_from_slice(&t.bytes_at(slot * per, per));
        }
        for t in self.peek_tokens(slot) {
            out.extend_from_slice(&t.to_le_bytes());
        }
    }

    /// The inverse of [`Self::append_slot_bytes`]: writes `bytes` back into `slot`,
    /// tensors and token history. The cold tier's hydrate path orders this before the
    /// slot is read, on the engine thread.
    pub fn restore_slot_bytes(&self, slot: usize, bytes: &[u8]) {
        assert!(slot < self.n_slots, "slot {slot} out of range");
        let per_bytes: Vec<usize> = self.tensors().map(|t| t.nb()[3]).collect();
        let need: usize = per_bytes.iter().sum::<usize>() + self.n_hist * 4;
        assert_eq!(bytes.len(), need, "a state snapshot must carry the whole slot");
        let mut at = 0usize;
        for (t, per) in self.tensors().zip(&per_bytes) {
            t.set_bytes_at(slot * per, &bytes[at..at + per]);
            at += per;
        }
        let hist: Vec<u32> = bytes[at..]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect();
        assert_eq!(hist.len(), self.n_hist, "the snapshot's history window");
        if self.n_hist > 0 {
            let mut all = self.tokens.lock().unwrap();
            all[slot * self.n_hist..][..self.n_hist].copy_from_slice(&hist);
        }
    }
}

#[cfg(test)]
mod tests;
