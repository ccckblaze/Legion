#!/usr/bin/env python3
# 4x DIFFERENT-context (head-salted => no shared prefix, true cold) C=4, like fw4k but adversarial.
import json, os, sys, threading, time, urllib.request
URL = os.environ.get("LB_URL", "http://127.0.0.1:8090")
KEY = os.environ.get("LB_API_KEY", "")
MODEL = os.environ.get("LB_MODEL", "qwen3.8-flash-next")
GEN = int(os.environ.get("LB_GEN", "256"))
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2600
FILL = ("Reimagine a lighthouse keeper logging nightly waves and distant ships; "
        "the ledger grows one careful paragraph at a time. ")
def _hdrs():
    h = {"Content-Type": "application/json"}
    if KEY: h["Authorization"] = "Bearer " + KEY
    return h
def post(ptext):
    body = {"model": MODEL, "messages": [{"role": "user", "content": ptext}],
            "max_tokens": GEN, "temperature": 0, "ignore_eos": True, "stream": False}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers=_hdrs())
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as x:
            d = json.loads(x.read().decode())
    except Exception as e:
        return {"err": type(e).__name__, "span": round(time.time()-t0,1)}
    u = d.get("usage") or {}; tm = d.get("timings") or {}
    ct = u.get("completion_tokens") or 0
    return {"ptok": u.get("prompt_tokens"), "ctok": ct,
            "tg": round(ct / max(1e-9, (time.time()-t0) - (tm.get("predicted_per_second") and 0 or 0)), 6),
            "tg_engine": tm.get("predicted_per_second"), "pp_engine": tm.get("prompt_per_second"),
            "span": round(time.time()-t0,1), "err": None}
def run():
    th = []; out = {}
    def work(i):
        # head salt in the FIRST 3 tokens -> every request is a different prefix, cold prefill for all
        p = "diffcase%02d %s" % (i, (FILL * R)[: 24 * R])
        out[i] = post(p)
    for i in range(4):
        t = threading.Thread(target=work, args=(i,)); th.append(t); t.start()
    for t in th: t.join()
    ok = [r for r in out.values() if r and not r.get("err")]
    tgs = sorted(round(r["ctok"]/r["span"],1) for r in ok)
    print(json.dumps({"reps": R, "wall": round(max((r["span"] for r in ok), default=0),1),
                      "n_ok": len(ok), "sum_tg": round(sum(tgs),1), "tg_list": tgs,
                      "ptok": ok[0]["ptok"] if ok else None}))
run()
