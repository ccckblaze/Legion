# reburst_max: precise full-window measure. The prompt length is *counted*, not estimated:
# /v1/messages/count_tokens fits body+filler to the target exactly, so phase B's prompt
# (phase A's text plus the instruction tail) lands on ctx-1 instead of overflowing it.
# usage: reburst_max.py <target_tokens> <C> <TAG> [--fit-only]
import json, os, sys, threading, time, urllib.error, urllib.request

BASE = os.environ.get("LB_URL", "http://127.0.0.1:8090")
KEY = os.environ.get("LB_API_KEY", "")
MODEL = os.environ.get("LB_MODEL", "qwen3.8-flash-next-test")
GEN = int(os.environ.get("LB_GEN_B", "256"))

SENT = ("Ancient lighthouses guided ships through fog with rotating mirrors. ",
        "The migratory geese navigated by stars and the earth magnetic field. ",
        "A cartographer drew coastlines from memory after a single voyage. ",
        "Deep-sea vessels logged pressure and temperature every hundred meters. ",
        "The signalman repeated each code twice to guard against static noise. ",
        "Winter keepers stocked oil lamps and spare wicks before the storms. ",
        "Harbor pilots memorized sandbars by the color of the breaking water. ",
        "The old surveyor measured bays with chains, knots and patient arithmetic. ",
        "Foghorn crews tested pitch weekly and logged every complaint by radio. ",
        "Tide tables printed in alphabetic order served the fishing fleet well. ",
        "Island postmen rowed mail between the lights on calm summer evenings. ",
        "The engineer rebuilt the stair inside the tower using granite pins only. ")
TAILS = (
    " Now count upward from 101 by ones, one number per space, for %d tokens.",
    " Now count downward from 900 by sevens, one number per space, for %d tokens.",
    " Now list the odd numbers starting at 101, one number per space, for %d tokens.",
    " Now list multiples of three starting at 303, one number per space, for %d tokens.",
    " Now count upward from 500 by threes, one number per space, for %d tokens.",
    " Now list numbers ending in 7 from 17 upward, one per space, for %d tokens.")
UNIT = " 7"  # fine-fit unit, about one token


def _req(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read().decode())


def count(text):
    return _req("/v1/messages/count_tokens",
                {"model": MODEL, "messages": [{"role": "user", "content": text}]})["input_tokens"]


def body(i, n):
    off, stride = i * 3 + 1, i + 2
    return "".join(SENT[(off + j * stride) % len(SENT)] for j in range(max(1, n)))


def fit(i, target, pre=""):
    """`pre` + body(i, n) + UNIT*k counted to exactly `target` (whole-prompt fitting: the stamp
    and the chat template are inside the measured number, so phase B's tail still fits ctx)."""
    text = lambda n, k: pre + body(i, n) + UNIT * k
    n = max(4, target // 13)
    for _ in range(10):
        d = target - count(text(n, 0))
        if abs(d) < 13 or n <= 1:
            break
        n = max(1, n + d // 13)
    while n > 1 and count(text(n, 0)) > target:
        n -= 1
    k = 0
    while count(text(n, k + 1)) <= target and k < 40:
        k += 1
    return text(n, k), n, k


def post(user_msg, gen):
    t0 = time.time()
    try:
        d = _req("/v1/chat/completions", {"model": MODEL, "messages": [
            {"role": "user", "content": user_msg}], "max_tokens": gen,
            "temperature": 0, "stream": False})
    except urllib.error.HTTPError as e:
        try:
            why = e.read().decode()[:170]
        except Exception:
            why = ""
        return {"err": "HTTP %s %s" % (e.code, why), "span": round(time.time() - t0, 1), "ctok": 0}
    except Exception as e:
        return {"err": str(e)[:170], "span": round(time.time() - t0, 1), "ctok": 0}
    u = d.get("usage", {})
    span = time.time() - t0
    ctok = u.get("completion_tokens", 0)
    det = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"err": None, "span": span, "ctok": ctok, "ptok": u.get("prompt_tokens", 0),
            "cached": det, "tg": (ctok / span if span > 0 and ctok else None)}


def run(target, C, tag, fit_only=False):
    stamp = int(time.time() * 1e6)
    prompts, fitinfo = [], []
    for i in range(C):
        tail = TAILS[i % len(TAILS)] % GEN
        tc = count("z" + tail) - count("z")
        pre = "dv%d-%d " % (i, stamp)
        t, n, k = fit(i, target - tc, pre)
        prompts.append(t)
        fitinfo.append({"i": i, "sent": n, "units": k, "tail": tc})
    counted = [count(p) for p in prompts]
    rec = {"tag": tag, "target": target, "C": C, "fit": fitinfo, "counted": counted}
    if fit_only:
        print(json.dumps(rec))
        return
    outA = {}

    def fill(i):
        outA[i] = post(prompts[i], 8)

    th = [threading.Thread(target=fill, args=(i,)) for i in range(C)]
    t0 = time.time()
    [t.start() for t in th]; [t.join() for t in th]
    wallA = time.time() - t0
    time.sleep(4)
    outB = {}

    def dec(i):
        outB[i] = post(prompts[i] + (TAILS[i % len(TAILS)] % GEN), GEN)

    th = [threading.Thread(target=dec, args=(i,)) for i in range(C)]
    t1 = time.time()
    [t.start() for t in th]; [t.join() for t in th]
    wallB = time.time() - t1
    okB = [r for r in outB.values() if not r["err"] and r.get("tg")]
    tgs = sorted(r["tg"] for r in okB)
    print(json.dumps({
        "tag": tag, "target": target, "counted": counted, "C": C,
        "wallA": round(wallA, 1), "wallB": round(wallB, 1),
        "fill_err": [r["err"] for r in outA.values() if r["err"]][:2],
        "dec_err": [r["err"] for r in outB.values() if r["err"]][:2],
        "tg_med_B": round(tgs[len(tgs) // 2], 3) if tgs else None,
        "tg_min_B": round(tgs[0], 3) if tgs else None,
        "tg_max_B": round(tgs[-1], 3) if tgs else None,
        "agg_B": round(sum(r["ctok"] for r in okB) / wallB, 2) if okB else None,
        "rows": [{"pt": r["ptok"], "ch": r.get("cached"), "ct": r["ctok"],
                  "tg": round(r["tg"], 2) if r.get("tg") else None,
                  "span": round(r["span"], 1)} for r in outB.values()]}))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    run(int(a[0]), int(a[1]), a[2], "--fit-only" in sys.argv)
