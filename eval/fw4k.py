#!/usr/bin/env python3
# 4x same-context (reps R) C=4, 2 rounds alternating (like fw4same): 4 seqs x R tokens resident.
import json, os, sys, threading, time, urllib.request
URL = os.environ.get("LB_URL", "http://127.0.0.1:8090")
KEY = os.environ.get("LB_API_KEY", "")
FILL = ("Reimagine a lighthouse keeper logging nightly waves and distant ships; "
        "the ledger grows one careful paragraph at a time. ")
LET = "abcdefgh"
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2600
def payload(salt):
    return salt + " " + FILL * R
def post(ptext):
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": ptext}],
            "max_tokens": 256, "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as x:
            d = json.loads(x.read().decode())
    except Exception as e:
        return {"err": type(e).__name__, "span": round(time.time() - t0, 1)}
    u = d.get("usage") or {}; tm = d.get("timings") or {}
    return {"ptok": u.get("prompt_tokens"), "ctok": u.get("completion_tokens"),
            "tg": tm.get("predicted_per_second"), "pp": tm.get("prompt_per_second"),
            "ttft": round(time.time() - t0, 1), "err": None}
res = {}
th = [threading.Thread(target=lambda i=i: res.__setitem__(i, post(payload(LET[i])))) for i in range(4)]
t0 = time.time()
for t in th: t.start()
for t in th: t.join()
wall = time.time() - t0
ok = [v for v in res.values() if not v.get("err")]
tgs = sorted(round(v["tg"], 1) for v in ok if v.get("tg"))
print(json.dumps({"reps": R, "wall": round(wall, 1), "n_ok": len(ok), "sum_tg": round(sum(tgs), 1),
                  "tg_list": tgs, "ptok": ok[0]["ptok"] if ok else None}))
