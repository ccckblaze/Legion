#!/usr/bin/env python3
"""v2 drift fixtures for --undeclared-tool-calls: pure-imitation and deep-context forms.

v1 (undecl_probe.py) taught drift with a forbid line + one example; the model now answers in
prose, so the coercion path never fires. v2 keeps only what production showed: the model copies
the assistant-turn markup (undeclared name) either directly, or after a long context prefix.
Verdicts/hashes only - no model text.
"""
import hashlib
import json
import os
import sys
import urllib.request

BASE = os.environ.get("LB_URL") or os.environ.get("PROBE_BASE") or "http://127.0.0.1:8097"
KEY = os.environ.get("LB_API_KEY") or os.environ.get("PROBE_KEY") or ""
LT, GT, EQ, NL = chr(60), chr(62), chr(61), chr(10)
FNO, PARO, PARC = LT + "function" + EQ, LT + "parameter" + EQ, LT + "/parameter" + GT
FNC = LT + "/function" + GT
MARKS = (LT + "function", LT + "/function", LT + "parameter", LT + "tool_call")

DECLARED = "weather_lookup"
TOOLS = [{
    "type": "function",
    "function": {
        "name": DECLARED,
        "description": "Look up the weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}]


def post(messages, max_tokens=400):
    body = {"model": "qwen3.8-flash-next", "messages": messages, "tools": TOOLS,
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    return ch.get("finish_reason"), msg, (msg.get("content") or "")


def sha8(t):
    return hashlib.sha256((t or "").encode()).hexdigest()[:8]


def verdict(tag, fr, msg, content):
    calls = msg.get("tool_calls") or []
    names = [((c.get("function") or {}).get("name") or "") for c in calls]
    drifted = [n for n in names if n != DECLARED]
    leak = any(m in content for m in MARKS)
    ok = bool(drifted) and not leak and fr == "tool_calls"
    args = ((calls[0].get("function") or {}) or {}).get("arguments") if drifted else ""
    try:
        keys = len(json.loads(args)) if (args or "").strip() else 0
    except Exception:
        keys = -1
    print("FIX %s %-14s fr=%-10s drifted=%d leak=%s keys=%d clen=%d %s"
          % (tag, sha8(".".join(names)), fr, len(drifted), leak, keys, len(content),
             "PASS" if ok else "FAIL"))
    return ok


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "call"
    target = "lookup"
    example = FNO + target + GT + NL + PARO + "city" + GT + NL + "Oslo" + NL + PARC + NL + FNC
    # A. pure imitation: the assistant turn shows the exact drifted form, ask for more of it.
    a_fr, a_msg, a_c = post([
        {"role": "system", "content": "Always answer with exactly one tool call, in the markup "
         "shown. Never write prose."},
        {"role": "user", "content": "What is the weather in Oslo?"},
        {"role": "assistant", "content": example},
        {"role": "user", "content": "What is the weather in Beijing? Same markup, same name."},
    ])
    okA = verdict("imitation", a_fr, a_msg, a_c)
    # B. deep context: 12K filler tokens before the same imitation pair (production drift shape).
    filler = ("The quarterly logistics report lists warehouse throughput, pallet cycles and "
              "dispatch windows in tabular prose. ") * 140
    b_fr, b_msg, b_c = post([
        {"role": "system", "content": "Always answer with exactly one tool call, in the markup "
         "shown. Never write prose."},
        {"role": "user", "content": filler + " Given the report above: what is the weather in Oslo?"},
        {"role": "assistant", "content": example},
        {"role": "user", "content": "And in Beijing? Same markup, same name."},
    ])
    okB = verdict("deepctx", b_fr, b_msg, b_c)
    print("RESULT %s v2 %s" % (mode, "ANYDRIFT" if (okA or okB) else "NODRIFT"))
    return 0


sys.exit(main())
