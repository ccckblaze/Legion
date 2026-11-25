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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// The tier's own layout version; a manifest of another format is ignored, not merged.
/// FORMAT 3: state snapshots carry the slot's token history window (and hydrate is
/// keyed by chains, C3).
pub const FORMAT: u32 = 3;

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

/// Read every manifest shard and count what is actually on the fly (a killed flusher can
/// leave an entry without a file: those drop out here rather than failing a request later).
///
/// Entries whose signature does not match this server's model are kept out of the map and
/// left on the disk for the model that owns them.
pub fn scan(cfg: &ConvDiskCfg, live: Option<&Signature>) -> (HashMap<String, BlockEntry>, Scanned) {
    let mut blocks: HashMap<String, BlockEntry> = HashMap::new();
    let (mut st_keep, mut bytes, mut missing, mut foreign) = (0usize, 0u64, 0usize, 0usize);
    let mut states: HashMap<String, StateEntry> = HashMap::new();
    let shards = fs_shards(&cfg.dir);
    for path in shards {
        let Ok(raw) = fs::read(&path) else { continue };
        let m = match serde_json::from_slice::<Manifest>(&raw) {
            Ok(m) if m.format == FORMAT => m,
            _ => continue,
        };
        for e in m.blocks {
            if e.chain.len() < 2 || e.bytes == 0 || e.tokens.is_empty() {
                missing += 1;
                continue;
            }
            if live.is_some_and(|l| l != &e.sig) {
                foreign += 1;
                continue;
            }
            if !blob_path(&cfg.dir, &e.chain).exists() {
                missing += 1;
                continue;
            }
            bytes += e.bytes;
            blocks.insert(e.chain.clone(), e);
        }
        for s in m.states {
            let keep = s.chain.len() >= 2
                && s.bytes > 0
                && live.map_or(true, |l| l == &s.sig)
                && state_path(&cfg.dir, &s.chain).exists();
            if keep {
                st_keep += 1;
                states.insert(s.chain.clone(), s);
            } else {
                missing += 1;
            }
        }
    }
    let found = blocks.len();
    (
        blocks,
        Scanned { blocks: found, states: st_keep, bytes, missing, foreign },
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
) -> Option<(ConvDiskCfg, Scanned)> {
    let cfg = ConvDiskCfg { dir: dir.to_path_buf(), gib, ttl_min, restore, verify_threads };
    for sub in ["blocks", "states", "pending"] {
        if let Err(e) = fs::create_dir_all(cfg.dir.join(sub)) {
            tracing::warn!("conv tier: {sub} under {}: {e}", cfg.dir.display());
            return None;
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
                Ok(m) if m.format == FORMAT => m,
                _ => continue,
            };
            for s in m.states {
                if s.bytes > 0 && s.sig == sig && state_path(&cfg.dir, &s.chain).exists() {
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
        if !jobs.is_empty() && self.inner.tx.send(Job::Blocks(jobs)).is_err() {
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
        let _ = self.inner.tx.send(Job::States(vec![job]));
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
    if let Some(p) = to.parent() {
        fs::create_dir_all(p)?;
    }
    if to.exists() {
        let _ = fs::remove_file(tmp);
        return Ok(());
    }
    fs::rename(tmp, to)
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
                        dump_block(&inner, &mut streams, j, &mut dirty);
                    }
                }
                Job::States(jobs) => {
                    for j in jobs {
                        dump_state(&inner, j, &mut dirty);
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
            fs::rename(&tmp, &to)
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
        self.buf_len = rest.len();
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
        };
        boot(&dir, 1.0, 0, true, 2).unwrap();
        let sig =
            Signature { model: "m".into(), digest: "d".into(), quant: "q".into(), block_bytes: 8, block_tokens: 64 };
        let chain = "ab1234";
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
}
