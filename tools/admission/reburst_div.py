# reburst_div: 异源稳态并发 decode 量尺 — 与 reburst_steady 同相位，但行行不同文
# Phase A: C 路同刻各灌一条 win 级异源链(max_tokens=8); Phase B: 续一条异源长任务(max_tokens=256)
import json, os, sys, time, threading, urllib.error, urllib.request

BASE = os.environ.get("LB_URL", "http://127.0.0.1:8090")
KEY = os.environ.get("LB_API_KEY", "")
GEN = int(os.environ.get("LB_GEN_B", "256"))
MODEL = os.environ.get("LB_MODEL", "qwen3.8-flash-next-test")

POOL = [
 "Ancient lighthouses guided ships through fog with rotating mirrors. ",
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
 "The engineer rebuilt the stair inside the tower using granite pins only. ",
]
TAILS = [
 " Now count upward from 101 by ones, one number per space, for %d tokens.",
 " Now count downward from 900 by sevens, one number per space, for %d tokens.",
 " Now list the odd numbers starting at 101, one number per space, for %d tokens.",
 " Now list multiples of three starting at 303, one number per space, for %d tokens.",
 " Now count upward from 500 by threes, one number per space, for %d tokens.",
 " Now list numbers ending in 7 from 17 upward, one per space, for %d tokens.",
]

TOKS_PER_SENT = 13.67  # 实测：POOL 句均 token 数（123K 格 ptok_med/句数 反推）

def body(i, win):
    # 行 i 用不同起点+不同步长遍历句子池 → 前缀逐行不同文；长度凑近 win-余量
    off, stride = i * 3 + 1, i + 2
    n = max(6, int((win - 900) / TOKS_PER_SENT))
    return "".join(POOL[(off + j * stride) % len(POOL)] for j in range(n))

def post(user_msg, gen):
    data = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": gen, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=data, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        # The engine's own refusal reason lives in the error body; without it a 400 is
        # unclassifiable from the client side.
        try:
            why = e.read().decode()[:200]
        except Exception:
            why = ""
        return {"err": "HTTP %s %s" % (e.code, why), "span": round(time.time() - t0, 1),
                "ctok": 0}
    except Exception as e:
        return {"err": str(e)[:180], "span": round(time.time() - t0, 1), "ctok": 0}
    u = d.get("usage", {})
    span = time.time() - t0
    ctok = u.get("completion_tokens", 0)
    return {"err": None, "span": span, "ctok": ctok, "ptok": u.get("prompt_tokens", 0),
            "tg": (ctok / span if span > 0 and ctok else None)}

def run(win, C, tag):
    stamp = int(time.time() * 1e6)
    prompts = ["dv%d-%d " % (i, stamp) + body(i, win) for i in range(C)]
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
    okB = [r for r in outB.values() if r and not r["err"]]
    tgs = sorted(r["tg"] for r in okB if r.get("tg"))
    ptoks = [r["ptok"] for r in okB]
    rec = {"tag": tag, "win": win, "C": C,
           "fill_errs": sum(1 for r in outA.values() if r["err"]),
           "dec_errs": C - len(okB),
           "wallA": round(wallA, 1), "wallB": round(wallB, 1),
           "tg_med_B": round(tgs[len(tgs)//2], 3) if tgs else None,
           "tg_min_B": round(tgs[0], 3) if tgs else None,
           "tg_max_B": round(tgs[-1], 3) if tgs else None,
           "agg_B": round(sum(r["ctok"] for r in okB) / wallB, 2) if okB else None,
           "ptok_med": int(sorted(ptoks)[len(ptoks)//2]) if ptoks else None}
    print(json.dumps(rec))

if __name__ == "__main__":
    win, C, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    run(win, C, tag)
