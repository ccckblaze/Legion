//! The conversation cold tier: KV blocks and recurrent-state snapshots kept on NVMe so a
//! restart or an eviction costs a read instead of a prefill storm (docs/ac922-cold-tier-l1.md).
//!
//! Entries are addressed by their token-chain hash: a block's `chain` is
//! `sha256(prompt[0..end])` over the little-endian token ids through the end of that block,
//! which fixes the block's contents (causal attention: the bytes are a function of the
//! prefix), names its file, and deduplicates: two sessions sharing a prompt share every
//! block of it, once on disk. `key` records `sha256(bytes)` for integrity and inspection.
//! State snapshots (oMLX's `GDNCheckpointEntry` shape) carry the chain of their save point.
//! Entries carry the compatibility signature (model digest, dims, quant, format), so an
//! entry only ever serves the model that wrote it, and a queued write lands in `pending/`
//! and is then linked into place, never over an existing file: keys are write-once, so a
//! superseded writer cannot replace a newer file (#5543). The size cap is global across
//! models, LRU-trimmed (#5497); the manifest restarts chains intact (#5255).
//!
//! The manifest is sharded by the chain's first byte (`manifest-<shard>.json`): a tier
//! holding hundreds of thousands of entries rewrites only the shards its flush touched.
//!
//! Blocks are gathered off the engine thread: `queue_blocks` hands the worker address
//! ranges (pool tensors outlive it), and the pool references the engine took at queue time
//! are returned through `drain_acks` once each block's file is on disk. A state's bytes
//! are copied on the engine thread (`bytes`): a saved slot is recycled on the next store.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// The tier's own layout version; a manifest of another format is ignored, not merged.
/// FORMAT 3: state snapshots carry the slot's token history window (and hydrate is
/// keyed by chains, C3).
/// FORMAT 4: entries carry the checkpoint's content fingerprint in their signature
/// (oMLX's key hygiene: a re-quantized pack must never serve from the old tier). A
/// format-3 manifest is still read, and its entries trusted once and re-signed with
/// the live fingerprint on their way into the index (they were written by this very
/// pack; the next flush rewrites the shard as v4).
pub const FORMAT: u32 = 4;
/// The last layout version whose entries carry no `pack` and are migrated on read.
pub const FORMAT_LEGACY: u32 = 3;

/// `--conv-disk`: the tier's directory, global size cap and retention. `None` (unset) is
/// the whole feature off; every other default matches the design doc.
#[derive(Clone, Debug)]
pub struct ConvDiskCfg {
    pub dir: PathBuf,
    pub gib: f64,
    pub ttl_min: u64,
    /// Hydrate at load (`--conv-disk-restore`, on): with it off the tier only ever writes.
    pub restore: bool,
    /// Read and hash blobs with this many workers while hydrating (`--conv-disk-verify-threads`):
    /// one thread hashes a full conversation's blocks in ~46 s, and it is the engine thread.
    pub verify_threads: usize,
    /// Spaced prompt checkpoints land on disk every this many save points, plus the last four
    /// of the prompt and every turn end (`--conv-disk-state-stride`; 0: dump every point).
    /// A full-window conversation keeps 32 states at stride 1 and 10 at stride 8, and a revive
    /// walks back to the deepest kept boundary either way (§368 C6).
    pub state_stride: usize,
    /// The writer queue's byte ceiling (`--conv-disk-pending-gib`, oMLX's bounded pending
    /// buffer): past it a dump steps aside - the blocks are acked back, the chain is left
    /// unclaimed, and a later completion or eviction queues it again. 0: no ceiling.
    pub pending_gib: f64,
}

impl ConvDiskCfg {
    /// The cap is global across models, oMLX #5497: one number bounds the whole directory.
    pub fn cap_bytes(&self) -> u64 {
        (self.gib * (1u64 << 30) as f64) as u64
    }

    pub fn pending_max(&self) -> u64 {
        (self.pending_gib * (1u64 << 30) as f64) as u64
    }
}

/// What a block must agree with before it may serve a request: anything that changes the
/// bytes of a block (a re-quantized pack, a different pool layout, this format).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Signature {
    pub model: String,
    /// The checkpoint's content fingerprint ([`pack_fingerprint`]): a re-quantized or
    /// replaced pack changes it, and its old blocks stop matching. Format-3 entries
    /// carry the empty string, which the read path migrates (see [`FORMAT`]).
    #[serde(default)]
    pub pack: String,
    pub digest: String,
    pub quant: String,
    pub block_bytes: u64,
    pub block_tokens: u32,
}

/// One stored KV block: the token-chain address, the hash of its bytes, the tokens of its
/// span (for debug), the previous block's chain, and the last time a request matched or
/// wrote it (the LRU clock).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockEntry {
    pub chain: String,
    pub key: String,
    pub prev: Option<String>,
    pub tokens: Vec<u32>,
    pub bytes: u64,
    pub sig: Signature,
    pub last_used_ms: u64,
}

/// A recurrent-state snapshot at a chain boundary (oMLX's `GDNCheckpointEntry`): keyed by
/// the chain at that point, taken when the engine saves a state (`--prefix-cache-interval`
/// points and block-aligned prompt ends).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateEntry {
    pub chain: String,
    pub key: String,
    pub tokens: u64,
    pub bytes: u64,
    pub sig: Signature,
    pub last_used_ms: u64,
}

/// One manifest shard, rewritten atomically when the tier adds or trims its entries.
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
    /// Format-3 entries trusted once and re-signed with the live pack fingerprint.
    pub migrated: usize,
}

/// What the tier actually gave back, the oMLX `cache/stats.py` counterpart: realized reuse
/// must be a number, not an impression. Relaxed adds, all of them on the engine thread.
#[derive(Default)]
pub struct ConvStats {
    pub walks: AtomicU64,
    pub chains_present: AtomicU64,
    pub blobs_ok: AtomicU64,
    pub blobs_mismatch: AtomicU64,
    pub states_ok: AtomicU64,
    pub states_bad: AtomicU64,
    pub alloc_fails: AtomicU64,
    pub upload_fails: AtomicU64,
    pub revive_aborts: AtomicU64,
    pub tokens_hydrated: AtomicU64,
    pub bytes_hydrated: AtomicU64,
    /// Dumps that stepped aside at the writer queue's ceiling (`--conv-disk-pending-gib`).
    pub dumps_backpressured: AtomicU64,
}

impl ConvStats {
    /// One hydrate walk over the chain: `present` block chains the tier had.
    pub fn note_walk(&self, present: u64) {
        self.walks.fetch_add(1, Ordering::Relaxed);
        self.chains_present.fetch_add(present, Ordering::Relaxed);
    }
    fn snapshot(&self) -> StatsSnapshot {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        StatsSnapshot {
            walks: g(&self.walks),
            chains_present: g(&self.chains_present),
            blobs_ok: g(&self.blobs_ok),
            blobs_mismatch: g(&self.blobs_mismatch),
            states_ok: g(&self.states_ok),
            states_bad: g(&self.states_bad),
            alloc_fails: g(&self.alloc_fails),
            upload_fails: g(&self.upload_fails),
            revive_aborts: g(&self.revive_aborts),
            tokens_hydrated: g(&self.tokens_hydrated),
            bytes_hydrated: g(&self.bytes_hydrated),
            dumps_backpressured: g(&self.dumps_backpressured),
            pending_bytes: 0,
        }
    }
}

/// `/health`'s view of [`ConvStats`].
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct StatsSnapshot {
    pub walks: u64,
    pub chains_present: u64,
    pub blobs_ok: u64,
    pub blobs_mismatch: u64,
    pub states_ok: u64,
    pub states_bad: u64,
    pub alloc_fails: u64,
    pub upload_fails: u64,
    pub revive_aborts: u64,
    pub tokens_hydrated: u64,
    pub bytes_hydrated: u64,
    pub dumps_backpressured: u64,
    /// Bytes in the writer queue right now (filled by `ConvTier::stats_snapshot`).
    pub pending_bytes: u64,
}

/// Hash-keyed blobs shard by the chain's first byte: one flat directory of tens of
/// thousands of files is the scalability wall oMLX solved with hash subdirectories.
pub fn blob_path(dir: &Path, chain: &str) -> PathBuf {
    let shard = chain.get(..2).unwrap_or("00");
    dir.join("blocks").join(shard).join(format!("{chain}.bin"))
}

/// State sidecars, same sharding.
pub fn state_path(dir: &Path, chain: &str) -> PathBuf {
    let shard = chain.get(..2).unwrap_or("00");
    dir.join("states").join(shard).join(format!("{chain}.bin"))
}

fn shard_name(shard: &str) -> String {
    format!("manifest-{shard}.json")
}

/// Signature match, with the format-3 migration: an entry that carries no `pack` and
/// otherwise agrees with the live model was written by this pack before the fingerprint
/// existed - it enters the index re-signed (and the next shard flush makes that durable).
/// The trust is scoped to the legacy layout: a v4 manifest with a `pack`-less entry is
/// corruption (or a downgrade attack), not history, and stays foreign.
fn sig_matches(e: &Signature, live: &Signature, legacy_layout: bool) -> bool {
    e == live || (legacy_layout && e.pack.is_empty() && live_legacy_eq(e, live))
}

fn live_legacy_eq(e: &Signature, live: &Signature) -> bool {
    e.model == live.model
        && e.digest == live.digest
        && e.quant == live.quant
        && e.block_bytes == live.block_bytes
        && e.block_tokens == live.block_tokens
}

/// Read every manifest shard and count what is actually on the fly (a killed flusher can
/// leave an entry without a file: those drop out here rather than failing a request later).
///
/// Entries whose signature does not match this server's model are kept out of the map and
/// left on the disk for the model that owns them.
pub fn scan(cfg: &ConvDiskCfg, live: Option<&Signature>) -> (HashMap<String, BlockEntry>, Scanned) {
    let mut blocks: HashMap<String, BlockEntry> = HashMap::new();
    let (mut st_keep, mut bytes, mut missing, mut foreign) = (0usize, 0u64, 0usize, 0usize);
    let mut migrated = 0usize;
    let mut states: HashMap<String, StateEntry> = HashMap::new();
    let shards = fs_shards(&cfg.dir);
    for path in shards {
        let Ok(raw) = fs::read(&path) else { continue };
        let m = match serde_json::from_slice::<Manifest>(&raw) {
            Ok(m) if m.format == FORMAT || m.format == FORMAT_LEGACY => m,
            _ => continue,
        };
        let legacy = m.format == FORMAT_LEGACY;
        for e in m.blocks {
            if e.chain.len() < 2 || e.bytes == 0 || e.tokens.is_empty() {
                missing += 1;
                continue;
            }
            if live.is_some_and(|l| !sig_matches(&e.sig, l, legacy)) {
                foreign += 1;
                continue;
            }
            if !blob_path(&cfg.dir, &e.chain).exists() {
                missing += 1;
                continue;
            }
            let mut e = e;
            if let Some(l) = live {
                if e.sig != *l {
                    e.sig = l.clone();
                    migrated += 1;
                }
            }
            bytes += e.bytes;
            blocks.insert(e.chain.clone(), e);
        }
        for s in m.states {
            let keep = s.chain.len() >= 2
                && s.bytes > 0
                && live.map_or(true, |l| sig_matches(&s.sig, l, legacy))
                && state_path(&cfg.dir, &s.chain).exists();
            if keep {
                st_keep += 1;
                let mut s = s;
                if let Some(l) = live {
                    if s.sig != *l {
                        s.sig = l.clone();
                        migrated += 1;
                    }
                }
                states.insert(s.chain.clone(), s);
            } else {
                missing += 1;
            }
        }
    }
    let found = blocks.len();
    (
        blocks,
        Scanned { blocks: found, states: st_keep, bytes, missing, foreign, migrated },
    )
}

fn fs_shards(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.file_name().map_or(false, |n| {
                let n = n.to_string_lossy();
                n.starts_with("manifest-") && n.ends_with(".json")
            }) {
                out.push(p);
            }
        }
    }
    out
}

/// `--conv-disk` given: make sure the tree exists and report what survived the last run.
/// C1 reads; C3 hydrates from the returned map.
pub fn boot(
    dir: &Path,
    gib: f64,
    ttl_min: u64,
    restore: bool,
    verify_threads: usize,
    state_stride: usize,
    pending_gib: f64,
) -> Option<(ConvDiskCfg, Scanned)> {
    let cfg = ConvDiskCfg {
        dir: dir.to_path_buf(),
        gib,
        ttl_min,
        restore,
        verify_threads,
        state_stride,
        pending_gib,
    };
    for sub in ["blocks", "states", "pending"] {
        if let Err(e) = fs::create_dir_all(cfg.dir.join(sub)) {
            tracing::warn!("conv tier: {sub} under {}: {e}", cfg.dir.display());
            return None;
        }
    }
    // A killed writer leaves `pending/` littered: those files were never linked in and the
    // scan ignores the directory, so boot clears it (oMLX's startup scan does the same).
    if let Ok(rd) = fs::read_dir(cfg.dir.join("pending")) {
        let mut gone = 0usize;
        for e in rd.flatten() {
            if e.path().is_file() && fs::remove_file(e.path()).is_ok() {
                gone += 1;
            }
        }
        if gone > 0 {
            tracing::info!("conv tier: cleared {gone} interrupted pending writes");
        }
    }
    let st = scan(&cfg, None).1;
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

/// Where a dumped slice lives: pinned host memory the worker may address directly, or one
/// device's VRAM, read out on the worker's own non-blocking stream.
#[derive(Clone, Copy, Debug)]
pub enum Mem {
    Host,
    Dev(u32),
}

/// A byte range of a live pool tensor. The engine thread builds these; the tensor's memory
/// outlives the tier (`Lm` owns the pools, the worker is joined before they drop).
#[derive(Clone, Copy, Debug)]
pub struct Slice {
    pub ptr: usize,
    pub len: usize,
    pub mem: Mem,
}

/// One prompt's finished blocks, in chain order, with the block ids whose pool references
/// the engine took at queue time (returned by [`ConvTier::drain_acks`] once written).
pub struct BlockJob {
    pub id: u32,
    pub slices: Vec<Slice>,
    pub tokens: Vec<u32>,
    pub chain: String,
    pub prev: Option<String>,
}

/// A saved state's bytes, copied on the engine thread (a saved slot is overwritten by the
/// next store), with the token count and chain of its save point.
pub struct StateJob {
    pub bytes: Vec<u8>,
    pub tokens: u64,
    pub chain: String,
}

enum Job {
    Blocks(Vec<BlockJob>),
    States(Vec<StateJob>),
}

/// The tier, shared between the engine loop (queue, match, ack) and its writer thread.
#[derive(Clone)]
pub struct ConvTier {
    inner: Arc<Inner>,
}

struct Inner {
    cfg: ConvDiskCfg,
    sig: Signature,
    tx: Sender<Job>,
    /// Block chains already on disk or in the queue: `queue_blocks` skips them, which is
    /// the tier's dedup (two sessions, one shared prompt) and its idempotence.
    chains: Mutex<HashSet<String>>,
    state_chains: Mutex<HashSet<String>>,
    /// Entries, by chain; the writer thread owns updates, the engine reads for hydrate.
    blocks: Mutex<HashMap<String, BlockEntry>>,
    states: Mutex<HashMap<String, StateEntry>>,
    /// Blocks whose files are on disk: the engine gives their pool references back.
    acks: Mutex<Vec<u32>>,
    bytes: Mutex<u64>,
    /// Bytes in the writer queue (materialized or promised): the backpressure gauge
    /// against `pending_max()` (oMLX's bounded pending buffer).
    pending: AtomicU64,
    stats: ConvStats,
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

impl ConvTier {
    /// Bring the tier up: seed the index from the boot scan and start the writer thread.
    /// The thread exits when the last `ConvTier` handle drops (`tx` closes), after its
    /// final manifest flush; `Lm` holds the only handle, so pool tensors outlive it.
    pub fn start(cfg: ConvDiskCfg, scanned: HashMap<String, BlockEntry>, sig: Signature) -> ConvTier {
        let (tx, rx) = mpsc::channel();
        let total: u64 = scanned.values().map(|e| e.bytes).sum();
        let mut state_chains = HashSet::new();
        let mut states = HashMap::new();
        // Re-read the state sidecars the boot scan counted (few of them).
        for path in fs_shards(&cfg.dir) {
            let Ok(raw) = fs::read(&path) else { continue };
            let m = match serde_json::from_slice::<Manifest>(&raw) {
                Ok(m) if m.format == FORMAT || m.format == FORMAT_LEGACY => m,
                _ => continue,
            };
            let legacy = m.format == FORMAT_LEGACY;
            for s in m.states {
                if s.bytes > 0 && sig_matches(&s.sig, &sig, legacy) && state_path(&cfg.dir, &s.chain).exists() {
                    let mut s = s;
                    if s.sig != sig {
                        s.sig = sig.clone();
                    }
                    state_chains.insert(s.chain.clone());
                    states.insert(s.chain.clone(), s);
                }
            }
        }
        let chains: HashSet<String> = scanned.keys().cloned().collect();
        let inner = Arc::new(Inner {
            cfg,
            sig,
            tx,
            chains: Mutex::new(chains),
            state_chains: Mutex::new(state_chains),
            blocks: Mutex::new(scanned),
            states: Mutex::new(states),
            acks: Mutex::new(Vec::new()),
            bytes: Mutex::new(total),
            pending: AtomicU64::new(0),
            stats: ConvStats::default(),
        });
        let me = inner.clone();
        std::thread::Builder::new()
            .name("conv-tier".into())
            .spawn(move || writer(me, rx))
            .expect("spawn conv-tier");
        ConvTier { inner }
    }

    pub fn sig(&self) -> &Signature {
        &self.inner.sig
    }

    pub fn cfg(&self) -> &ConvDiskCfg {
        &self.inner.cfg
    }

    pub fn stats(&self) -> &ConvStats {
        &self.inner.stats
    }

    pub fn stats_snapshot(&self) -> StatsSnapshot {
        let mut s = self.inner.stats.snapshot();
        s.pending_bytes = self.inner.pending.load(Ordering::Relaxed);
        s
    }

    /// Queue a prompt's finished blocks. Chains already stored or queued are skipped (the
    /// ids of blocks skipped this way are acked immediately: their references were taken
    /// for the queue and nothing else is pending on them).
    pub fn queue_blocks(&self, mut jobs: Vec<BlockJob>) {
        {
            let mut chains = self.inner.chains.lock().unwrap();
            jobs.retain(|j| {
                if chains.contains(&j.chain) {
                    self.inner.acks.lock().unwrap().push(j.id);
                    false
                } else {
                    chains.insert(j.chain.clone());
                    true
                }
            });
        }
        if jobs.is_empty() {
            return;
        }
        // Backpressure (oMLX's capped pending buffer): past the ceiling this dump steps
        // aside whole - ids ack straight back, chains stay unclaimed, and a later prompt
        // completion or eviction queues the blocks again. The engine thread never waits.
        let n: u64 = jobs.iter().flat_map(|j| j.slices.iter().map(|s| s.len as u64)).sum();
        let cap = self.inner.cfg.pending_max();
        if cap > 0 && self.inner.pending.fetch_add(n, Ordering::Relaxed) + n > cap {
            self.inner.pending.fetch_sub(n, Ordering::Relaxed);
            self.inner.stats.dumps_backpressured.fetch_add(jobs.len() as u64, Ordering::Relaxed);
            let mut acks = self.inner.acks.lock().unwrap();
            for j in &jobs {
                acks.push(j.id);
            }
            return;
        }
        if self.inner.tx.send(Job::Blocks(jobs)).is_err() {
            self.inner.pending.fetch_sub(n, Ordering::Relaxed);
            tracing::warn!("conv tier: writer gone, block dumps dropped");
        }
    }

    pub fn queue_state(&self, job: StateJob) {
        {
            let mut chains = self.inner.state_chains.lock().unwrap();
            if chains.contains(&job.chain) {
                return;
            }
            chains.insert(job.chain.clone());
        }
        let n = job.bytes.len() as u64;
        let cap = self.inner.cfg.pending_max();
        if cap > 0 && self.inner.pending.fetch_add(n, Ordering::Relaxed) + n > cap {
            self.inner.pending.fetch_sub(n, Ordering::Relaxed);
            self.inner.state_chains.lock().unwrap().remove(&job.chain);
            self.inner.stats.dumps_backpressured.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if self.inner.tx.send(Job::States(vec![job])).is_err() {
            self.inner.pending.fetch_sub(n, Ordering::Relaxed);
        }
    }

    /// The ids whose files are on disk (or dropped): the engine decrefs their blocks.
    pub fn drain_acks(&self) -> Vec<u32> {
        std::mem::take(&mut *self.inner.acks.lock().unwrap())
    }

    /// The entry serving `chain`, if the tier has one whose blob is on disk.
    pub fn block(&self, chain: &str) -> Option<BlockEntry> {
        let e = self.inner.blocks.lock().unwrap().get(chain).cloned();
        e.filter(|_| blob_path(&self.inner.cfg.dir, chain).exists())
    }

    /// Whether the chain is stored or already queued: the engine skips building the
    /// slices of blocks the tier would drop at `queue_blocks` anyway.
    pub fn have(&self, chain: &str) -> bool {
        self.inner.chains.lock().unwrap().contains(chain)
    }

    pub fn state(&self, chain: &str) -> Option<StateEntry> {
        let e = self.inner.states.lock().unwrap().get(chain).cloned();
        e.filter(|_| state_path(&self.inner.cfg.dir, chain).exists())
    }

    /// A match or a hydrate touched these chains: move them to the LRU head.
    pub fn touch(&self, chains: &[String]) {
        let now = now_ms();
        let mut b = self.inner.blocks.lock().unwrap();
        for c in chains {
            if let Some(e) = b.get_mut(c) {
                e.last_used_ms = now;
            }
        }
        let mut s = self.inner.states.lock().unwrap();
        for c in chains {
            if let Some(e) = s.get_mut(c) {
                e.last_used_ms = now;
            }
        }
    }
}

/// Read `job`'s slices into one buffer: host ranges by direct copy, device ranges on the
/// worker's non-blocking stream (the engine's streams never wait on us, and its writes to
/// a dumped block were ordered before queueing: logits had been read for that step).
fn gather(slices: &[Slice], streams: &mut HashMap<u32, *mut std::ffi::c_void>) -> Option<Vec<u8>> {
    let total: usize = slices.iter().map(|s| s.len).sum();
    let mut out = vec![0u8; total];
    let mut at = 0usize;
    let mut used: Vec<u32> = Vec::new();
    for s in slices {
        let dst = out.as_mut_ptr().cast::<u8>().wrapping_add(at);
        match s.mem {
            Mem::Host => unsafe {
                std::ptr::copy_nonoverlapping(s.ptr as *const u8, dst, s.len);
            },
            #[cfg(feature = "cuda")]
            Mem::Dev(d) => {
                let stream = device_stream(d, streams)?;
                unsafe {
                    cuda::cudaSetDevice(d as std::os::raw::c_int);
                    let e = cuda::cudaMemcpyAsync(
                        dst.cast(),
                        s.ptr as *const std::ffi::c_void,
                        s.len,
                        cuda::D2H,
                        stream,
                    );
                    if e != 0 {
                        tracing::warn!("conv tier: d2h error {e}");
                        return None;
                    }
                }
                if !used.contains(&d) {
                    used.push(d);
                }
            }
            #[cfg(not(feature = "cuda"))]
            Mem::Dev(_) => return None,
        }
        at += s.len;
    }
    #[cfg(feature = "cuda")]
    for d in used {
        if let Some(s) = streams.get(&d) {
            unsafe {
                cuda::cudaSetDevice(d as std::ffi::c_int);
                let e = cuda::cudaStreamSynchronize(*s);
                if e != 0 {
                    tracing::warn!("conv tier: stream sync {e}");
                    return None;
                }
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = &used;
    Some(out)
}

#[cfg(feature = "cuda")]
fn device_stream(d: u32, streams: &mut HashMap<u32, *mut std::ffi::c_void>) -> Option<*mut std::ffi::c_void> {
    if let Some(s) = streams.get(&d) {
        return Some(*s);
    }
    let mut s: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe {
        cuda::cudaSetDevice(d as std::os::raw::c_int);
        // cudaStreamNonBlocking = 1: never serializes with the engine's work.
        if cuda::cudaStreamCreateWithFlags(&mut s, 1) != 0 {
            tracing::warn!("conv tier: no stream on device {d}");
            return None;
        }
    }
    streams.insert(d, s);
    Some(s)
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::os::raw::{c_int, c_uint, c_void};

    pub const D2H: c_int = 4;

    unsafe extern "C" {
        pub fn cudaSetDevice(d: c_int) -> c_int;
        pub fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            n: usize,
            kind: c_int,
            s: *mut c_void,
        ) -> c_int;
        pub fn cudaStreamCreateWithFlags(s: *mut *mut c_void, flags: c_uint) -> c_int;
        pub fn cudaStreamSynchronize(s: *mut c_void) -> c_int;
    }
}

/// Link a freshly written file into place: never over an existing one (#5543).
fn link_once(tmp: &Path, to: &Path) -> std::io::Result<()> {
    let Some(p) = to.parent() else { return Ok(()) };
    fs::create_dir_all(p)?;
    if to.exists() {
        let _ = fs::remove_file(tmp);
        return Ok(());
    }
    // oMLX's durability order (`_fsync_parent_dir`): the manifest may name a file only
    // once the bytes and the directory entry are both on the platter.
    sync_file(tmp);
    fs::rename(tmp, to)?;
    sync_dir(p);
    Ok(())
}

fn sync_file(path: &Path) {
    if let Ok(f) = fs::File::open(path) {
        let _ = f.sync_all();
    }
}

fn sync_dir(dir: &Path) {
    if let Ok(d) = fs::File::open(dir) {
        let _ = d.sync_all();
    }
}

fn writer(inner: Arc<Inner>, rx: mpsc::Receiver<Job>) {
    let mut streams: HashMap<u32, *mut std::ffi::c_void> = HashMap::new();
    let mut dirty: HashSet<String> = HashSet::new();
    let mut last_flush = Instant::now();
    loop {
        let first = match rx.recv_timeout(Duration::from_millis(1000)) {
            Ok(j) => j,
            Err(RecvTimeoutError::Disconnected) => break,
            Err(_) => {
                if last_flush.elapsed() > Duration::from_secs(5) && !dirty.is_empty() {
                    flush(&inner, &mut dirty);
                    last_flush = Instant::now();
                }
                continue;
            }
        };
        let mut batch = vec![first];
        batch.extend(rx.try_iter());
        for job in batch {
            match job {
                Job::Blocks(jobs) => {
                    for j in jobs {
                        let n: u64 = j.slices.iter().map(|s| s.len as u64).sum();
                        dump_block(&inner, &mut streams, j, &mut dirty);
                        inner.pending.fetch_sub(n, Ordering::Relaxed);
                    }
                }
                Job::States(jobs) => {
                    for j in jobs {
                        let n = j.bytes.len() as u64;
                        dump_state(&inner, j, &mut dirty);
                        inner.pending.fetch_sub(n, Ordering::Relaxed);
                    }
                }
            }
        }
        trim(&inner, &mut dirty);
        if !dirty.is_empty() && last_flush.elapsed() > Duration::from_secs(2) {
            flush(&inner, &mut dirty);
            last_flush = Instant::now();
        }
    }
    flush(&inner, &mut dirty);
    #[cfg(feature = "cuda")]
    for (_, s) in streams {
        unsafe {
            cuda::cudaStreamSynchronize(s);
        }
    }
}

fn dump_block(
    inner: &Inner,
    streams: &mut HashMap<u32, *mut std::ffi::c_void>,
    j: BlockJob,
    dirty: &mut HashSet<String>,
) {
    let ack = |inner: &Inner| inner.acks.lock().unwrap().push(j.id);
    let Some(bytes) = gather(&j.slices, streams) else {
        inner.chains.lock().unwrap().remove(&j.chain);
        return ack(inner);
    };
    let key = sha256_hex(&bytes);
    let dir = &inner.cfg.dir;
    let tmp = dir.join("pending").join(format!("{}.bin", &j.chain));
    let to = blob_path(dir, &j.chain);
    if fs::write(&tmp, &bytes).and_then(|_| link_once(&tmp, &to)).is_err() {
        tracing::warn!("conv tier: block write failed");
        inner.chains.lock().unwrap().remove(&j.chain);
        return ack(inner);
    }
    let n = bytes.len() as u64;
    let entry = BlockEntry {
        chain: j.chain.clone(),
        key,
        prev: j.prev,
        tokens: j.tokens,
        bytes: n,
        sig: inner.sig.clone(),
        last_used_ms: now_ms(),
    };
    inner.blocks.lock().unwrap().insert(j.chain.clone(), entry);
    *inner.bytes.lock().unwrap() += n;
    dirty.insert(j.chain[..2].to_string());
    ack(inner);
}

fn dump_state(inner: &Inner, j: StateJob, dirty: &mut HashSet<String>) {
    let key = sha256_hex(&j.bytes);
    let dir = &inner.cfg.dir;
    let tmp = dir.join("pending").join(format!("s-{}.bin", &j.chain));
    let to = state_path(dir, &j.chain);
    if fs::write(&tmp, &j.bytes).and_then(|_| link_once(&tmp, &to)).is_err() {
        tracing::warn!("conv tier: state write failed");
        inner.state_chains.lock().unwrap().remove(&j.chain);
        return;
    }
    let n = j.bytes.len() as u64;
    let entry = StateEntry {
        chain: j.chain.clone(),
        key,
        tokens: j.tokens,
        bytes: n,
        sig: inner.sig.clone(),
        last_used_ms: now_ms(),
    };
    inner.states.lock().unwrap().insert(j.chain.clone(), entry);
    *inner.bytes.lock().unwrap() += n;
    dirty.insert(j.chain[..2].to_string());
}

/// The cap is the hard wall (oMLX #5497): drop least-recently-used entries, blocks first,
/// until the tier fits again. `--conv-disk-ttl-min` drops aged entries the same way.
fn trim(inner: &Inner, dirty: &mut HashSet<String>) {
    let cap = inner.cfg.cap_bytes();
    let ttl = inner.cfg.ttl_min;
    let age_ms = if ttl > 0 { Some((ttl as u64).saturating_mul(60_000)) } else { None };
    let over = *inner.bytes.lock().unwrap() > cap;
    if !over && age_ms.is_none() {
        return;
    }
    let now = now_ms();
    let mut victims: Vec<String> = Vec::new();
    {
        let b = inner.blocks.lock().unwrap();
        let mut v: Vec<(&String, u64)> = b.iter().map(|(k, e)| (k, e.last_used_ms)).collect();
        v.sort_by_key(|&(_, t)| t);
        let mut held = *inner.bytes.lock().unwrap();
        for (k, t) in v {
            if (!over || held <= cap) && age_ms.map_or(true, |a| now.saturating_sub(t) <= a) {
                break;
            }
            if let Some(e) = b.get(k) {
                held = held.saturating_sub(e.bytes);
            }
            victims.push(k.clone());
        }
    }
    for k in &victims {
        if let Some(e) = inner.blocks.lock().unwrap().remove(k) {
            *inner.bytes.lock().unwrap() -= e.bytes;
            let _ = fs::remove_file(blob_path(&inner.cfg.dir, &e.chain));
            dirty.insert(k[..2].to_string());
        }
        inner.chains.lock().unwrap().remove(k);
    }
    let mut svictims: Vec<String> = Vec::new();
    {
        let s = inner.states.lock().unwrap();
        let mut v: Vec<(&String, u64)> = s.iter().map(|(k, e)| (k, e.last_used_ms)).collect();
        v.sort_by_key(|&(_, t)| t);
        for (k, t) in v {
            if age_ms.map_or(true, |a| now.saturating_sub(t) <= a) {
                break;
            }
            svictims.push(k.clone());
        }
    }
    for k in &svictims {
        if let Some(e) = inner.states.lock().unwrap().remove(k) {
            *inner.bytes.lock().unwrap() -= e.bytes;
            let _ = fs::remove_file(state_path(&inner.cfg.dir, &e.chain));
            dirty.insert(k[..2].to_string());
        }
        inner.state_chains.lock().unwrap().remove(k);
    }
}

/// Rewrite each dirty shard atomically (tmp + rename), grouping both entry kinds by chain.
fn flush(inner: &Inner, dirty: &mut HashSet<String>) {
    let shards = dirty.clone();
    dirty.clear();
    for shard in shards {
        let m = Manifest {
            format: FORMAT,
            blocks: inner
                .blocks
                .lock()
                .unwrap()
                .values()
                .filter(|e| e.chain.starts_with(&shard))
                .cloned()
                .collect(),
            states: inner
                .states
                .lock()
                .unwrap()
                .values()
                .filter(|e| e.chain.starts_with(&shard))
                .cloned()
                .collect(),
        };
        let to = inner.cfg.dir.join(shard_name(&shard));
        let tmp = inner.cfg.dir.join(format!(".tmp-{}", shard_name(&shard)));
        let json = match serde_json::to_vec(&m) {
            Ok(j) => j,
            Err(_) => continue,
        };
        if fs::write(&tmp, &json).and_then(|_| {
            let f = fs::File::open(&tmp)?;
            f.sync_all()?;
            drop(f);
            fs::rename(&tmp, &to)?;
            sync_dir(&inner.cfg.dir);
            Ok(())
        }).is_err() {
            tracing::warn!("conv tier: manifest write failed");
        }
    }
}

/// Chain addresses: `sha256` of the little-endian token ids through each whole block's end,
/// one snapshot per block a prompt of `tokens` length would have. Cheap enough to run on
/// the engine thread per prompt (`4 bytes × tokens` hashed once, streamed).
pub fn block_chains(tokens: &[u32], block: usize) -> Vec<String> {
    let mut h = Sha256::new();
    let mut out = Vec::with_capacity(tokens.len() / block);
    for chunk in tokens.chunks(block) {
        let mut le = Vec::with_capacity(chunk.len() * 4);
        for t in chunk {
            le.extend_from_slice(&t.to_le_bytes());
        }
        h.update(&le);
        if chunk.len() == block {
            out.push(sha256_hex_fin(&h));
        }
    }
    out
}

/// The chain of one exact prefix: the address of a state saved after `tokens`.
pub fn chain_of(tokens: &[u32]) -> String {
    let mut le = Vec::with_capacity(tokens.len() * 4);
    for t in tokens {
        le.extend_from_slice(&t.to_le_bytes());
    }
    sha256_hex(&le)
}

/// The count of leading whole blocks whose chains `accept` takes, with the same
/// streaming hash as [`Self::block_chains`]: the hydrate walk stops at the first
/// block the tier does not have, so an unknown prompt costs one hash update per
/// block only while the run holds.
pub fn chain_run(tokens: &[u32], block: usize, mut accept: impl FnMut(&str) -> bool) -> usize {
    let mut h = Sha256::new();
    let mut n = 0usize;
    for chunk in tokens.chunks(block) {
        if chunk.len() != block {
            break;
        }
        let mut le = Vec::with_capacity(chunk.len() * 4);
        for t in chunk {
            le.extend_from_slice(&t.to_le_bytes());
        }
        h.update(&le);
        let chain = sha256_hex_fin(&h);
        if !accept(&chain) {
            break;
        }
        n += 1;
    }
    n
}

/// The checkpoint's content fingerprint, oMLX's key hygiene (format/seed/dims in the
/// key, never a false hit) taken one step further: per model file - a split GGUF's
/// shards or every shard in a directory - `sha256(name || size || mtime || head ||
/// tail)`, the file digests then hashed in path order into one hex. A full hash of a
/// ~70 GiB pack per boot is not affordable; a re-quantized or replaced pack is: the
/// GGUF metadata header it rewrites sits in the head, and its data moves the tail.
pub fn pack_fingerprint(path: &Path) -> String {
    let mut files: Vec<PathBuf> = Vec::new();
    if path.is_dir() {
        if let Ok(rd) = fs::read_dir(path) {
            for e in rd.flatten() {
                let p = e.path();
                let ext = p.extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default();
                if (ext == "gguf" || ext == "safetensors") && p.is_file() {
                    files.push(p);
                }
            }
        }
    } else if path.is_file() {
        files.push(path.to_path_buf());
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if let Some(stem) = shard_stem(&name) {
            if let Some(dir) = path.parent() {
                if let Ok(rd) = fs::read_dir(dir) {
                    for e in rd.flatten() {
                        let p = e.path();
                        let n = p.file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
                        if n != name && p.is_file() && shard_stem(&n).as_deref() == Some(stem.as_str()) {
                            files.push(p);
                        }
                    }
                }
            }
        }
    }
    files.sort();
    let mut outer = Sha256::new();
    for f in &files {
        let name = f.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let md = f.metadata().ok();
        let size = md.as_ref().map_or(0, |m| m.len());
        let mtime = md
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        let mut h = Sha256::new();
        h.update(name.as_bytes());
        h.update(&size.to_le_bytes());
        h.update(&mtime.to_le_bytes());
        h.update(&sample(f, size, false));
        h.update(&sample(f, size, true));
        outer.update(sha256_hex_fin(&h).as_bytes());
    }
    sha256_hex_fin(&outer)
}

/// `model-00001-of-00004.gguf` -> `model` (`None` when the name is not a split shard's).
fn shard_stem(name: &str) -> Option<String> {
    let (head, ext) = name.rsplit_once('.')?;
    if ext != "gguf" && ext != "safetensors" {
        return None;
    }
    let (a, b) = head.rsplit_once("-of-")?;
    if b.is_empty() || !b.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let (stem, i) = a.rsplit_once('-')?;
    if stem.is_empty() || i.is_empty() || !i.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(stem.to_string())
}

/// One edge sample of a model file: at most 64 KiB of its head or tail.
fn sample(path: &Path, size: u64, tail: bool) -> Vec<u8> {
    let want = size.min(1 << 16) as usize;
    if want == 0 {
        return Vec::new();
    }
    let mut f = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut buf = vec![0u8; want];
    use std::io::{Read, Seek, SeekFrom};
    let from = if tail { size.saturating_sub(want as u64) } else { 0 };
    if f.seek(SeekFrom::Start(from)).is_err() {
        return Vec::new();
    }
    match f.read(buf.as_mut_slice()) {
        Ok(0) => Vec::new(),
        Ok(n) => {
            buf.truncate(n);
            buf
        }
        Err(_) => Vec::new(),
    }
}

/// A blob's bytes, only if they hash to `key`: the chain is the *address*, and the
/// key check is what makes a stale, corrupted or other-model blob a miss rather
/// than poisoned KV. Reads on the engine thread during placement.
pub fn read_verified(dir: &Path, chain: &str, key: &str, want_bytes: u64) -> Option<Vec<u8>> {
    let bytes = fs::read(blob_path(dir, chain)).ok()?;
    if bytes.len() as u64 != want_bytes || sha256_hex(&bytes) != key {
        return None;
    }
    Some(bytes)
}

/// The same check for a state sidecar.
pub fn read_state_verified(dir: &Path, chain: &str, key: &str, want_bytes: u64) -> Option<Vec<u8>> {
    let bytes = fs::read(state_path(dir, chain)).ok()?;
    if bytes.len() as u64 != want_bytes || sha256_hex(&bytes) != key {
        return None;
    }
    Some(bytes)
}

/// Hydrate verifies in windows of at most this many blocks: the window bounds the transient
/// RAM (512 blocks of ~1 MiB) and the engine thread still uploads in order, window by window.
pub const VERIFY_WINDOW: usize = 512;

/// A window's blobs, each read and hashed by a worker pool, returned in the order of `items`:
/// per blob the same verdict as [`read_verified`], with the engine thread waiting on the pool
/// instead of doing the hashing itself. Workers touch only the filesystem and pure hashing;
/// a `None` at position i means the run stops at i exactly as the serial loop would.
pub fn read_verified_window(dir: &Path, items: &[(&str, String, u64)], threads: usize) -> Vec<Option<Vec<u8>>> {
    let mut out: Vec<Option<Vec<u8>>> = items.iter().map(|_| None).collect();
    if items.is_empty() {
        return out;
    }
    let threads = threads.clamp(1, items.len());
    let chunk = items.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (group, part) in out.chunks_mut(chunk).zip(items.chunks(chunk)) {
            scope.spawn(move || {
                for (slot, (chain, key, want)) in group.iter_mut().zip(part) {
                    *slot = read_verified(dir, chain, key, *want);
                }
            });
        }
    });
    out
}

/// What one conversation's hydrate needs, decided by walking the index (D1): the run's end,
/// the deepest boundary with a length-matched state, and the blobs in between, chain-ordered.
/// `plan` is the same walk `place()` ran inline before: a task takes it once, the engine
/// re-plans when there is no task yet.
#[derive(Clone, Debug)]
pub struct HydratePlan {
    /// The block the resume stands on: blobs start at `chains[start_pre]` in `block_chains`.
    pub start: usize,
    /// The chain run found on disk, capped by the prompt (for the walk counter and logs).
    pub m: usize,
    /// The chains to hydrate, ascending: `block_chains(prompt[..m*block])[start..j]`.
    pub chains: Vec<String>,
    /// The boundary state: (chain of the resume point, key, bytes); `None`: stateless model.
    pub state: Option<(String, String, u64)>,
}

/// Where a hydrate plan's walk stopped failing, so the engine can bump the same counters
/// the inline path did (state reads and blob reads have different ledgers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HydrateFault {
    /// The state's bytes mismatched or vanished between plan and read.
    State,
    /// A blob's sha mismatched after it was indexed.
    Blob,
    /// A blob or state the plan named was no longer there (a trim race): not a fault of
    /// integrity, the extension just aborts.
    Gone,
}

/// One step's consumption from a running [`HydrateTask`] ([`HydrateTask::take`]).
pub struct HydrateDelivery {
    /// Verified blobs, chain-ordered (possibly empty while the producer is behind).
    pub blobs: Vec<Vec<u8>>,
    /// The state's bytes, delivered exactly once, after the last blob.
    pub state: Option<Vec<u8>>,
    /// Every blob and the state have been delivered: the engine may extend its table.
    pub done: bool,
    /// The plan died (a trim race or an integrity failure): abort, bump the fault's counter.
    pub failed: Option<HydrateFault>,
}

/// The read+verify part of a hydrate, run ahead of the engine (D1): a producer thread reads
/// and sha-verifies the plan's windows while the engine keeps stepping, and each `place()`
/// attempt uploads what is ready, up to its budget. Cross-conversation hydrations overlap
/// each other and the decode windows: before this, a burst of revive-heavy requests paid
/// every session's full verify+upload serially on the engine thread (§366, §369).
/// Pool blocks uploaded between steps are parked in the task, so a `Wait` retry loses
/// nothing; when the job dies the engine sweeps the parked blocks back (abort + take_ids).
pub struct HydrateTask {
    plan: HydratePlan,
    dir: PathBuf,
    threads: usize,
    /// Verified windows the engine has not taken: bounds the transient RAM at
    /// `cap * VERIFY_WINDOW * block_bytes` per task (2 x 512 x ~1.2 MiB ~ 1.2 GiB).
    cap: usize,
    inner: Mutex<HydrateInner>,
    room: std::sync::Condvar,
}

struct HydrateInner {
    windows: std::collections::VecDeque<Vec<Vec<u8>>>,
    produced: usize,
    taken: usize,
    state: Option<Vec<u8>>,
    done: bool,
    failed: Option<HydrateFault>,
    aborted: bool,
    /// Pool blocks the engine parked after uploading them, pending the table's extension.
    ids: Vec<u32>,
}

impl HydrateTask {
    /// The walk shared by `place()` (re-plan) and [`Self::start`] (one-shot): `None` when
    /// there is nothing to hydrate (misaligned resume, no chain run, no state match).
    pub fn plan(
        tier: &ConvTier,
        prompt: &[u32],
        resumed: usize,
        block: usize,
        state_size: Option<u64>,
    ) -> Option<HydratePlan> {
        let len = prompt.len();
        let start = resumed / block;
        let cap = (len - 1) / block;
        if resumed % block != 0 || start >= cap {
            return None;
        }
        let m = chain_run(prompt, block, |c| tier.block(c).is_some()).min(cap).max(start);
        tier.stats().note_walk(m as u64);
        if m <= start {
            return None;
        }
        let chains = block_chains(&prompt[..m * block], block);
        let j = match state_size {
            None => m,
            Some(size) => (start + 1..=m)
                .rev()
                .find(|&i| tier.state(&chains[i - 1]).is_some_and(|e| e.bytes == size))
                .unwrap_or(start),
        };
        if j <= start {
            return None;
        }
        let state = match &state_size {
            None => None,
            Some(size) => tier
                .state(&chains[j - 1])
                .map(|e| (chains[j - 1].clone(), e.key, *size)),
        };
        Some(HydratePlan {
            start,
            m,
            chains: chains[start..j].to_vec(),
            state,
        })
    }

    /// Spawn the producer thread for one plan. `tier` is cloned (it is an `Arc` shell):
    /// the thread needs the index (a plan can race a trim between entries) and the directory.
    pub fn start(tier: &ConvTier, plan: HydratePlan) -> Arc<Self> {
        let task = Arc::new(Self {
            dir: tier.cfg().dir.clone(),
            threads: tier.cfg().verify_threads,
            cap: 2,
            plan,
            inner: Mutex::new(HydrateInner {
                windows: std::collections::VecDeque::new(),
                produced: 0,
                taken: 0,
                state: None,
                done: false,
                failed: None,
                aborted: false,
                ids: Vec::new(),
            }),
            room: std::sync::Condvar::new(),
        });
        let producer = Arc::clone(&task);
        let tier = tier.clone();
        std::thread::spawn(move || producer.produce(&tier));
        task
    }

    fn produce(self: Arc<Self>, tier: &ConvTier) {
        let fail = |t: &Self, f: HydrateFault| {
            let mut i = t.inner.lock().unwrap();
            i.failed = Some(f);
            i.done = true;
            t.room.notify_all();
        };
        if let Some((chain, key, bytes)) = self.plan.state.clone() {
            match read_state_verified(&self.dir, &chain, &key, bytes) {
                Some(b) => self.inner.lock().unwrap().state = Some(b),
                None if tier.state(&chain).is_none() => return fail(&self, HydrateFault::Gone),
                None => {
                    tracing::warn!("conv tier: revive aborted, state sidecar verification failed for {chain}");
                    return fail(&self, HydrateFault::State);
                }
            }
        }
        let mut at = 0usize;
        while at < self.plan.chains.len() {
            let end = (at + VERIFY_WINDOW).min(self.plan.chains.len());
            let mut items: Vec<(&str, String, u64)> = Vec::with_capacity(end - at);
            for chain in &self.plan.chains[at..end] {
                match tier.block(chain) {
                    Some(e) => items.push((chain.as_str(), e.key, e.bytes)),
                    None => return fail(&self, HydrateFault::Gone),
                }
            }
            let got = read_verified_window(&self.dir, &items, self.threads);
            let mut blobs = Vec::with_capacity(end - at);
            for g in got {
                match g {
                    Some(b) => blobs.push(b),
                    None => return fail(&self, HydrateFault::Blob),
                }
            }
            let mut i = self.inner.lock().unwrap();
            while i.windows.len() >= self.cap && !i.aborted && i.failed.is_none() {
                i = self.room.wait(i).unwrap();
            }
            if i.aborted || i.failed.is_some() {
                return;
            }
            i.produced += blobs.len();
            i.windows.push_back(blobs);
            drop(i);
            at = end;
        }
        let mut i = self.inner.lock().unwrap();
        i.done = true;
        self.room.notify_all();
    }

    /// Take up to `budget` verified blobs, chain-ordered, plus the state when this drains
    /// the plan. Never blocks: an empty take just means the producer is behind.
    pub fn take(&self, budget: usize) -> HydrateDelivery {
        let mut i = self.inner.lock().unwrap();
        if let Some(f) = i.failed {
            return HydrateDelivery { blobs: Vec::new(), state: None, done: false, failed: Some(f) };
        }
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        while blobs.len() < budget {
            match i.windows.front_mut() {
                Some(w) if !w.is_empty() => {
                    blobs.push(w.remove(0));
                    i.taken += 1;
                }
                Some(_) => {
                    i.windows.pop_front();
                }
                None => break,
            }
        }
        let done = i.done && i.windows.is_empty() && i.taken == i.produced;
        let state = if done { i.state.take() } else { None };
        self.room.notify_all();
        HydrateDelivery { blobs, state, done, failed: None }
    }

    pub fn plan_ref(&self) -> &HydratePlan {
        &self.plan
    }

    /// Park this step's uploaded pool blocks so a `Wait` retry resumes from them.
    pub fn park_ids(&self, ids: &[u32]) {
        self.inner.lock().unwrap().ids.extend_from_slice(ids);
    }

    /// The engine's or the sweep's: the parked blocks leave the task for good.
    pub fn take_ids(&self) -> Vec<u32> {
        std::mem::take(&mut self.inner.lock().unwrap().ids)
    }

    /// Stop the producer (a dying job, or a plan aborted mid-drain).
    pub fn abort(&self) {
        let mut i = self.inner.lock().unwrap();
        i.aborted = true;
        self.room.notify_all();
    }
}

/// SHA-256, self-contained (FIPS 180-4): the tier's only hash, for addresses and integrity.
#[derive(Clone)]
pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total: u64,
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

impl Sha256 {
    pub fn new() -> Self {
        Self {
            h: [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                0x1f83d9ab, 0x5be0cd19],
            buf: [0u8; 64],
            buf_len: 0,
            total: 0,
        }
    }

    fn compress(&mut self, block: &[u8]) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, v) in self.h.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *x = x.wrapping_add(v);
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let buf = self.buf;
                self.compress(&buf);
                self.buf_len = 0;
            }
        }
        let full = data.len() / 64 * 64;
        for block in data[..full].chunks_exact(64) {
            self.compress(block);
        }
        let rest = &data[full..];
        self.buf[..rest.len()].copy_from_slice(rest);
        // Add, never overwrite: when a small tail exhausted `data` inside the partial-block
        // branch above, `buf_len` already counts those bytes (a plain `=` here dropped them,
        // and `pad` then hashed over them: `sha256(b"abc")` ignored `abc`). Inputs that
        // enter aligned (the chain's 256 B block updates, whole-block blob keys) never
        // took the broken path, which is why the on-disk tier is self-consistent today.
        self.buf_len += rest.len();
    }

    pub fn finish(mut self) -> [u8; 32] {
        self.pad();
        self.h.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<u8>>().try_into().unwrap()
    }

    fn pad(&mut self) {
        let bits = self.total * 8;
        self.update(&[0x80]);
        while self.buf_len != 56 {
            if self.buf_len > 56 {
                let buf = self.buf;
                self.compress(&buf);
                self.buf = [0u8; 64];
                self.buf_len = 0;
            } else {
                self.buf[self.buf_len] = 0;
                self.buf_len += 1;
            }
        }
        // The length field must not re-enter `update` (it would re-pad): append raw.
        self.buf[56..].copy_from_slice(&bits.to_be_bytes());
        let buf = self.buf;
        self.compress(&buf);
    }
}

fn sha256_hex_fin(h: &Sha256) -> String {
    h.clone().finish_hex()
}

impl Sha256 {
    fn finish_hex(self) -> String {
        self.finish().iter().map(|b| format!("{b:02x}")).collect()
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    h.finish_hex()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // A 1e6-byte-ish multi-block input, the FIPS stream case (also covers the pad path
        // whose buffer crosses a block boundary).
        let big = vec![b'a'; 1000];
        assert_eq!(
            sha256_hex(&big),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }

    #[test]
    fn chains_snapshot_per_block() {
        let t: Vec<u32> = (0..200).collect();
        let c = block_chains(&t, 64);
        assert_eq!(c.len(), 3); // 192 tokens of whole blocks
        assert_eq!(c[0], chain_of(&t[..64]));
        assert_eq!(c[2], chain_of(&t[..192]));
        assert_ne!(c[0], c[1]);
    }

    #[test]
    fn scan_drops_missing_and_counts_foreign() {
        let dir = std::env::temp_dir().join(format!("ct-test-{}", uuid::Uuid::new_v4()));
        let cfg = ConvDiskCfg {
            dir: dir.clone(),
            gib: 1.0,
            ttl_min: 0,
            restore: true,
            verify_threads: 2,
            state_stride: 8,
            pending_gib: 64.0,
        };
        boot(&dir, 1.0, 0, true, 2, 8, 64.0).unwrap();
        let sig =
            Signature { model: "m".into(), pack: "p".into(), digest: "d".into(), quant: "q".into(), block_bytes: 8, block_tokens: 64 };
        let chain = "ab1234";
        fs::create_dir_all(dir.join("blocks").join("ab")).unwrap();
        fs::write(blob_path(&dir, chain), [7u8; 8]).unwrap();
        let m = Manifest {
            format: FORMAT,
            blocks: vec![
                BlockEntry {
                    chain: chain.into(),
                    key: "k".into(),
                    prev: None,
                    tokens: vec![1; 64],
                    bytes: 8,
                    sig: sig.clone(),
                    last_used_ms: 0,
                },
                BlockEntry {
                    chain: "ff0001".into(),
                    key: "g".into(),
                    prev: None,
                    tokens: vec![2; 64],
                    bytes: 8,
                    sig: sig.clone(),
                    last_used_ms: 0,
                },
                BlockEntry {
                    chain: "cd0002".into(),
                    key: "h".into(),
                    prev: None,
                    tokens: vec![3; 64],
                    bytes: 8,
                    sig: Signature { model: "other".into(), ..sig.clone() },
                    last_used_ms: 0,
                },
            ],
            states: Vec::new(),
        };
        fs::write(dir.join(shard_name("ab")), serde_json::to_vec(&m).unwrap()).unwrap();
        let (map, st) = scan(&cfg, Some(&sig));
        assert_eq!(map.len(), 1);
        assert!(map.contains_key(chain));
        assert_eq!((st.blocks, st.missing, st.foreign), (1, 1, 1));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn chain_run_agrees_and_stops() {
        let toks: Vec<u32> = (0..200).collect();
        let all = block_chains(&toks, 64);
        assert_eq!(all.len(), 3);
        // All present: the run counts every whole block.
        assert_eq!(chain_run(&toks, 64, |c| all.iter().any(|x| x == c)), 3);
        // The second chain missing: the run stops at one, with identical strings.
        assert_eq!(chain_run(&toks, 64, |c| c != all[1] && all.contains(&c.to_string())), 1);
        // A short tail chunk never joins the run.
        assert_eq!(chain_run(&toks[..192], 64, |_| true), 3);
        assert_eq!(chain_run(&toks[..192], 64, |_| false), 0);
    }

    /// A format-3 manifest (no `pack` in its signatures) enters the index re-signed with
    /// the live fingerprint - the tier survives the upgrade - while a live-pack mismatch
    /// is still foreign, and a v4 entry from another pack never matches.
    #[test]
    fn scan_migrates_v3_and_gates_by_pack() {
        let dir = std::env::temp_dir().join(format!("ct-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(dir.join("blocks").join("ab")).unwrap();
        let cfg = ConvDiskCfg {
            dir: dir.clone(),
            gib: 1.0,
            ttl_min: 0,
            restore: true,
            verify_threads: 2,
            state_stride: 8,
            pending_gib: 0.0,
        };
        let legacy = Signature {
            model: "m".into(),
            pack: String::new(),
            digest: "d".into(),
            quant: "q".into(),
            block_bytes: 8,
            block_tokens: 64,
        };
        let live = Signature { pack: "p-live".into(), ..legacy.clone() };
        let other = Signature { pack: "p-other".into(), ..legacy.clone() };
        fs::write(dir.join("blocks").join("ab").join("ab1234.bin"), [7u8; 8]).unwrap();
        let m = Manifest {
            format: FORMAT_LEGACY,
            blocks: vec![
                BlockEntry {
                    chain: "ab1234".into(),
                    key: "k".into(),
                    prev: None,
                    tokens: vec![1, 2],
                    bytes: 8,
                    sig: legacy.clone(),
                    last_used_ms: 1,
                },
                BlockEntry {
                    chain: "ab5678".into(),
                    key: "k".into(),
                    prev: None,
                    tokens: vec![3, 4],
                    bytes: 8,
                    sig: other.clone(),
                    last_used_ms: 1,
                },
            ],
            states: Vec::new(),
        };
        fs::write(dir.join(shard_name("ab")), serde_json::to_vec(&m).unwrap()).unwrap();
        let (blocks, st) = scan(&cfg, Some(&live));
        assert_eq!(blocks.len(), 1);
        assert_eq!(st.migrated, 1);
        assert_eq!(st.foreign, 1);
        assert_eq!(blocks["ab1234"].sig, live);
        // A v4 manifest indexes only live-pack entries: a different pack is foreign, and
        // an empty pack (which only a v3 writer ever produced) is no longer trusted.
        let mut v4ok = m.blocks[0].clone();
        v4ok.sig = live.clone();
        let v4other = m.blocks[1].clone();
        let mut v4blank = m.blocks[0].clone();
        v4blank.chain = "ab4321".into();
        let m4 = Manifest {
            format: FORMAT,
            blocks: vec![v4ok, v4other, v4blank],
            states: Vec::new(),
        };
        fs::write(dir.join(shard_name("ab")), serde_json::to_vec(&m4).unwrap()).unwrap();
        let (blocks, st) = scan(&cfg, Some(&live));
        assert_eq!(blocks.len(), 1);
        assert_eq!(st.migrated, 0);
        assert_eq!(st.foreign, 2);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The fingerprint follows the file's bytes where the signature must: head or tail
    /// changes move it, an unrelated rename of the same bytes does not (name is hashed,
    /// so a rename DOES move it - by design: the path is part of the model's identity).
    #[test]
    fn pack_fingerprint_tracks_content() {
        let dir = std::env::temp_dir().join(format!("ct-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let f = dir.join("model-00001-of-00001.gguf");
        fs::write(&f, vec![1u8; 1 << 18]).unwrap();
        let a = pack_fingerprint(&f);
        assert!(!a.is_empty());
        let mut bytes = vec![1u8; 1 << 18];
        bytes[(1 << 18) - 8..].copy_from_slice(&[9u8; 8][..]);
        fs::write(&f, &bytes).unwrap();
        let b = pack_fingerprint(&f);
        assert_ne!(a, b, "a tail change must move the fingerprint");
        // Split shards all count: touching a sibling moves the whole fingerprint.
        fs::write(&f, vec![1u8; 1 << 18]).unwrap();
        let s = dir.join("model-00002-of-00002.gguf");
        fs::write(&s, vec![3u8; 1 << 17]).unwrap();
        let c = pack_fingerprint(&f);
        fs::write(&s, vec![4u8; 1 << 17]).unwrap();
        let d = pack_fingerprint(&f);
        assert_ne!(c, d, "a shard change must move the fingerprint");
        assert_eq!(shard_stem("model-00001-of-00002.gguf").as_deref(), Some("model"));
        assert_eq!(shard_stem("plain.gguf"), None);
        let _ = fs::remove_dir_all(&dir);
    }
}
