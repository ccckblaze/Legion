#!/usr/bin/env python3
# engine-agnostic quality gate: det x2 on 6 fixed prompts + accuracy checks + optional 123K needle.
import hashlib, json, os, random, re, sys, time, urllib.request
URL = os.environ.get("LB_URL", "http://127.0.0.1:8090")
KEY = os.environ.get("LB_API_KEY", "")
def ask(pt, mt=512, timeout=2400):
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": pt}],
            "max_tokens": mt, "temperature": 0}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"x-api-key": KEY} if KEY else {})})
    with urllib.request.urlopen(req, timeout=timeout) as x:
        d = json.loads(x.read().decode())
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), (d.get("usage") or {}).get("completion_tokens", 0)
Q = [("What is 27 times 43? Reply with just the number.", r"1161"),
     ("A train covers 96 km in 48 minutes. What is its speed in km/h? Reply with just the number.", r"\b120\b"),
     ("The Arkwright contract was signed in 1873 by Elias Thorn. In what year was the contract signed? Reply with just the year.", r"1873"),
     ("How many whole hours are in 500 minutes? Reply with just the number.", r"\b8\b"),
     ("Complete exactly one line of Python after def fib(n): that returns fib(n-1)+fib(n-2) for n>1, using a base case n<2 returning n. Reply only that line.", None),
     ("Which is bigger: 7/8 or 5/6? Reply with just the fraction.", r"7/8")]
mode = sys.argv[1] if len(sys.argv) > 1 else "det"
if mode == "needle":
    seed = random.randint(100000000, 999999999)
    unit = ("Reimagine a lighthouse keeper logging nightly waves and distant ships; "
            "the ledger grows one careful paragraph at a time. ")
    cut = 2500
    txt = unit * cut + f" The vault code for shelf {seed} is filed under exactly nine digits: {seed}. " + unit * cut
    t0 = time.time()
    a, ctok = ask(txt + " Question: what is the nine-digit vault code? Reply with just the number.", mt=512)
    print(json.dumps({"needle_ok": str(seed) in a, "units": cut*2+1,
                      "span": round(time.time() - t0, 1)}))
    sys.exit(0)
shas = []
for rnd in (1, 2):
    got = []
    for q, _ in Q:
        a, _ = ask(q)
        got.append(hashlib.sha256(a.encode()).hexdigest()[:16])
    shas.append(got)
acc = 0
for q, pat in Q:
    a, _ = ask(q)
    if pat is None or re.search(pat, a):
        acc += 1
print(json.dumps({"det_same": shas[0] == shas[1], "acc_pass": acc, "acc_of": len(Q), "shas": shas[0]}))
