//! The conversation cold tier: KV blocks and recurrent-state snapshots kept on NVMe so a
//! restart or an eviction costs a read instead of a prefill storm (docs/ac922-cold-tier-l1.md).
//!
//! Storage is content-addressed per KV block, oMLX's `PagedSSDCacheManager` shape: a block's
//! key is the sha256 of its bytes, and its index entry carries the compatibility signature
//! (model digest, dims, quant, format), so an entry only ever serves a byte-identical block of
//! the same model, and blocks two sessions share are stored once. Recurrent state does not
//! slice, so it travels as a sidecar per chain boundary, oMLX's `GDNCheckpointIndex` shape.
//! A queued write lands in `pending/` and is then linked into place, never over an existing
//! file: keys are write-once, so a superseded writer cannot replace a newer file (#5543).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The tier's own layout version; a manifest of another format is ignored, not merged.
pub const FORMAT: u32 = 1;

/// `--conv-disk`: the tier's directory, global size cap and retention. `None` (unset) is
/// the whole feature off; every other default matches the design doc.
#[derive(Clone, Debug)]
pub struct ConvDiskCfg {
    pub dir: PathBuf,
    pub gib: f64,
    pub ttl_min: u64,
    /// Hydrate at load (`--conv-disk-restore`, on): with it off the tier only ever writes.
    pub restore: bool,
}

impl ConvDiskCfg {
    /// The cap is global across models, oMLX #5497: one number bounds the whole directory.
    pub fn cap_bytes(&self) -> u64 {
        (self.gib * (1u64 << 30) as f64) as u64
    }
}

/// What a block must agree with before it may serve a request: anything that changes the
/// bytes of a block (a re-quantized pack, a different pool layout, this format).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Signature {
    pub model: String,
    pub digest: String,
    pub quant: String,
    pub block_bytes: u64,
    pub block_tokens: u32,
}

/// One stored KV block: its bytes' hash, its span of prompt tokens, and the previous block's
/// hash, so chains rebuild at scan without walking the request that made them (#5255: a
/// restart that half-registers a long shared prefix pays the cold storm again).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockEntry {
    pub key: String,
    pub prev: Option<String>,
    pub tokens: Vec<u32>,
    pub bytes: u64,
    pub sig: Signature,
    pub last_used_ms: u64,
}

/// A recurrent-state snapshot at a chain boundary (oMLX's `GDNCheckpointEntry`): keyed by the
/// block-chain hash at that point, taken at every `--prefix-cache-interval` save.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateEntry {
    pub key: String,
    pub chain: String,
    pub tokens: u64,
    pub bytes: u64,
    pub sig: Signature,
    pub last_used_ms: u64,
}

/// `dir/manifest.json`, rewritten atomically when the tier adds or trims entries.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub format: u32,
    pub blocks: Vec<BlockEntry>,
    pub states: Vec<StateEntry>,
}

/// What a startup scan found, for the load-time log line and `/health` later.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scanned {
    pub blocks: usize,
    pub states: usize,
    pub bytes: u64,
    pub missing: usize,
    pub foreign: usize,
}

/// Hash-keyed blobs shard by the key's first byte: one flat directory of tens of thousands
/// of files is the scalability wall oMLX solved with hash subdirectories.
pub fn blob_path(dir: &Path, key: &str) -> PathBuf {
    let shard = key.get(..2).unwrap_or("00");
    dir.join("blocks").join(shard).join(format!("{key}.bin"))
}

/// State sidecars, same sharding.
pub fn state_path(dir: &Path, key: &str) -> PathBuf {
    let shard = key.get(..2).unwrap_or("00");
    dir.join("states").join(shard).join(format!("{key}.bin"))
}

/// Read the manifest and count what is actually on the fly (a killed flusher can leave a
/// manifest entry without a file: those drop out here rather than failing a request later).
///
/// The returned map is by key, ready for the chain walk; `Scanned::foreign` counts entries
/// whose signature stopped matching this server's model (kept out of the map, left on the
/// disk for the model that owns them).
pub fn scan(cfg: &ConvDiskCfg, live: Option<&Signature>) -> (HashMap<String, BlockEntry>, Scanned) {
    let mut blocks = HashMap::new();
    let (mut st_count, mut bytes, mut missing, mut foreign) = (0usize, 0u64, 0usize, 0usize);
    let manifest = match fs::read(cfg.dir.join("manifest.json")) {
        Ok(raw) => serde_json::from_slice::<Manifest>(&raw).unwrap_or_default(),
        Err(_) => Manifest::default(),
    };
    if manifest.format != FORMAT {
        return (blocks, Scanned::default());
    }
    for e in manifest.blocks {
        if e.sig.block_bytes == 0 || e.tokens.is_empty() {
            missing += 1;
            continue;
        }
        if live.is_some_and(|l| l != &e.sig) {
            foreign += 1;
            continue;
        }
        if !blob_path(&cfg.dir, &e.key).exists() {
            missing += 1;
            continue;
        }
        bytes += e.bytes;
        blocks.insert(e.key.clone(), e);
    }
    for s in manifest.states {
        if s.sig.block_bytes == 0
            || live.is_some_and(|l| l != &s.sig)
            || !state_path(&cfg.dir, &s.key).exists()
        {
            missing += 1;
        } else {
            st_count += 1;
        }
    }
    (
        blocks,
        Scanned { blocks: blocks.len(), states: st_count, bytes, missing, foreign },
    )
}

/// `--conv-disk` given: make sure the tree exists and report what survived the last run.
/// C1 only reads; C3 hydrates from the returned map.
pub fn boot(dir: &Path, gib: f64, ttl_min: u64, restore: bool) -> Option<(ConvDiskCfg, Scanned)> {
    let cfg = ConvDiskCfg { dir: dir.to_path_buf(), gib, ttl_min, restore };
    for sub in ["blocks", "states", "pending"] {
        if let Err(e) = fs::create_dir_all(cfg.dir.join(sub)) {
            tracing::warn!("conv tier: {sub} under {}: {e}", cfg.dir.display());
            return None;
        }
    }
    let (_, st) = scan(&cfg, None);
    tracing::info!(
        "conv tier: {} blocks {} states {:.1} GiB on disk ({} missing, {} foreign), restore {}",
        st.blocks,
        st.states,
        st.bytes as f64 / (1u64 << 30) as f64,
        st.missing,
        st.foreign,
        if cfg.restore { "on" } else { "off" },
    );
    Some((cfg, st))
}
