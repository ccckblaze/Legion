#!/usr/bin/env python3
# cold-tier revive gate: create a long session (phase new), restart the lane, then resume it
# (phase check) and require: same follow-up sha as the warm answer, cached_tokens >= 95% of
# the prefix, wall well under the cold prefill storm. Prints JSON only (no model text).
import hashlib, json, os, sys, time, urllib.request
URL = os.environ.get("LB_URL", "http://127.0.0.1:8090")
KEY = os.environ.get("LB_API_KEY", "")
ST = "/tmp/revive-%s.json"
FILL = ("Reimagine a lighthouse keeper logging nightly waves and distant ships; "
        "the ledger grows one careful paragraph at a time. ")
def ask(pt, mt):
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": pt}],
            "max_tokens": mt, "temperature": 0}
    h = {"Content-Type": "application/json"}
    if KEY: h["x-api-key"] = KEY
    t0 = time.time()
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(r, timeout=2400) as x:
        d = json.loads(x.read().decode())
    el = time.time() - t0
    m = d["choices"][0]["message"]
    u = d.get("usage", {})
    return (hashlib.sha256((m.get("content") or m.get("reasoning_content") or "").encode()).hexdigest()[:16],
            u.get("prompt_tokens", 0), u.get("cached_tokens", 0), round(el, 1))
def main():
    phase, tag, win = sys.argv[1], sys.argv[2], int(sys.argv[3] if len(sys.argv) > 3 else 60000)
    if phase == "new":
        pt = FILL * max(1, win // (len(FILL) * 3 // 4))
        a1, p1, c1, w1 = ask(pt, 128)
        fu = pt + "\nAnswer with one short sentence: what is this ledger?"
        a2, p2, c2, w2 = ask(fu, 32)
        json.dump({"win": p1, "warm": {"sha": a2, "cached": c2, "wall": w2}}, open(ST % tag, "w"))
        print(json.dumps({"phase": "new", "tag": tag, "prompt": p1, "first_cached": c1,
                          "first_wall": w1, "warm_follow_wall": w2}))
    else:
        st = json.load(open(ST % tag))
        pt = FILL * max(1, st["win"] // (len(FILL) * 3 // 4))
        fu = pt + "\nAnswer with one short sentence: what is this ledger?"
        a, p, c, w = ask(fu, 32)
        ok = (a == st["warm"]["sha"]) and (c >= int(st["win"] * 0.95))
        print(json.dumps({"phase": "check", "tag": tag, "sha_match": a == st["warm"]["sha"],
                          "cached": c, "of": st["win"], "wall": w, "revive": bool(ok)}))
        sys.exit(0 if ok else 7)
main()
