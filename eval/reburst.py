#!/usr/bin/env python3
# Correct C-concurrency classes: TOTAL=2*C requests, but at most C in flight (worker pool).
# Mirrors the official deterministic client (8 prompts x2 rounds at C=1/4/16, max-seqs=16).
import json, os, sys, threading, time, queue, urllib.request

URL = os.environ.get("LB_URL", "http://127.0.0.1:8090")
KEY = os.environ.get("LB_API_KEY", "")
MODEL = os.environ.get("LB_MODEL", "qwen3.8-flash-next")
GEN = int(os.environ.get("LB_GEN", "256"))
FILL = ("Reimagine a lighthouse keeper logging nightly waves and distant ships; "
        "the ledger grows one careful paragraph at a time. ")

def _hdrs():
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = "Bearer " + KEY
    return h

def post(ptext, gen):
    body = {"model": MODEL, "messages": [{"role": "user", "content": ptext}],
            "max_tokens": gen, "temperature": 0, "stream": False}
    if os.environ.get("LB_IGNORE_EOS", "0") == "1":
        body["ignore_eos"] = True
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=_hdrs())
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=2400) as x:
            d = json.loads(x.read().decode())
    except Exception as e:
        return {"err": type(e).__name__, "span": round(time.time() - t0, 1)}
    u = d.get("usage") or {}
    tm = d.get("timings") or {}
    return {"ptok": u.get("prompt_tokens"), "ctok": u.get("completion_tokens"),
            "tg": tm.get("predicted_per_second"), "err": None, "span": round(time.time() - t0, 1)}

def classrun(base, C, tag, outfile, win, nreq0=None):
    nreq = 2 * C if nreq0 is None else nreq0
    q = queue.Queue()
    for i in range(nreq):
        q.put(i)
    out = {}
    def worker():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            out[i] = post(base + ("request-ordinal-%d marker. " % (i + 1) if os.environ.get("LB_SALT", "0") == "1" else ""), GEN)
    t0 = time.time()
    th = [threading.Thread(target=worker) for _ in range(C)]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    ok = [r for r in out.values() if r and not r["err"]]
    tgs = sorted(r["tg"] for r in ok if r.get("tg"))
    rec = {"kind": "C%d" % C, "tag": tag, "win": win, "n": nreq,
           "wall": round(wall, 1), "agg": round(sum(r["ctok"] for r in ok) / wall, 1) if ok else None,
           "tg_med": tgs[len(tgs)//2] if tgs else None, "errs": nreq - len(ok)}
    line = json.dumps(rec)
    print(line, flush=True)
    open(outfile, "a").write(line + "\n")

if __name__ == "__main__":
    W = int(sys.argv[1]); C = int(sys.argv[2]); tag = sys.argv[3]
    outfile = "/data/work/legion/logs/reburst-%s.jsonl" % tag
    base = FILL * max(1, int((W - GEN) / 24))
    classrun(base, C, tag, outfile, W)
