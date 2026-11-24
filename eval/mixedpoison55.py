#!/usr/bin/env python3
# Poison-pill test: 8x 8K steady decode streams, optionally +1 cold 262K fired at t=30s.
# Reports per-8K-stream tg and aggregate, with/without the long intrusion in same boot.
import json, os, sys, threading, time, urllib.request
URL = os.environ.get("LB_URL", "http://127.0.0.1:8090")
KEY = os.environ.get("LB_API_KEY", "")
FILL = ("Reimagine a lighthouse keeper logging nightly waves and distant ships; "
        "the ledger grows one careful paragraph at a time. ")
def body(pt, mt):
    return {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": pt}],
            "max_tokens": mt, "temperature": 0, "ignore_eos": True}
def post(b, out):
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=3600) as x:
            d = json.loads(x.read().decode())
        tm = d.get("timings") or {}
        out.append({"tg": tm.get("predicted_per_second"), "pp": tm.get("prompt_per_second"),
                    "ctok": (d.get("usage") or {}).get("completion_tokens"), "span": round(time.time()-t0,1), "err": None})
    except Exception as e:
        out.append({"err": type(e).__name__, "span": round(time.time()-t0,1)})
def wave8k(mt, salt):
    out = []
    ths = []
    pt = " ".join(salt + str(i) + " " + FILL * 332 for i in range(1))
    for i in range(8):
        p = "%s%d " % (salt, i) + FILL * 332
        b = body(p, mt)
        t = threading.Thread(target=post, args=(b, out)); ths.append(t)
    t0 = time.time()
    for t in ths: t.start()
    return ths, out, t0
mode = sys.argv[1] if len(sys.argv) > 1 else "base"
if mode == "base":
    ths, out, t0 = wave8k(1024, "B")
    for t in ths: t.join()
    wall = time.time() - t0
    ok = [v for v in out if not v.get("err")]
    tgs = sorted(round(v["tg"],1) for v in ok if v.get("tg"))
    print(json.dumps({"mode":"base","wall":round(wall,1),"n":len(ok),
                      "agg":round(sum(v["ctok"] for v in ok)/wall,1),"tg_list":tgs}))
else:
    ths, out, t0 = wave8k(1024, "P")
    longout = []
    def firelong():
        time.sleep(55)
        p = "L " + FILL * 10890
        post(body(p, 128), longout)
    lt = threading.Thread(target=firelong); lt.start()
    for t in ths: t.join()
    lt.join()
    wall = time.time() - t0
    ok = [v for v in out if not v.get("err")]
    tgs = sorted(round(v["tg"],1) for v in ok if v.get("tg"))
    l = longout[0] if longout else {}
    print(json.dumps({"mode":"poison","wall":round(wall,1),"n":len(ok),
                      "agg":round(sum(v["ctok"] for v in ok)/wall,1),"tg_list":tgs,
                      "long":{"tg":round(l.get("tg") or 0,1),"pp":round(l.get("pp") or 0,1),"span":l.get("span")}}))
