"""Check an agintent units jsonl against the spec and print the summary for its README.

Usage: .venv/bin/python -I scripts/agintent_check_units_20261024.py <units.jsonl> [n_examples]
"""
import collections, json, re, statistics, sys

path = sys.argv[1]
n_ex = int(sys.argv[2]) if len(sys.argv) > 2 else 2
units = [json.loads(l) for l in open(path, encoding="utf-8")]
ids = collections.Counter(u["unit_id"] for u in units)
assert all(c == 1 for c in ids.values()), [i for i, c in ids.items() if c > 1]
byid = {u["unit_id"]: u for u in units}
ban = re.compile(r"summitbridge|novastar", re.I)
lens = []
for u in units:
    assert re.fullmatch(r"[a-z0-9_-]+", u["unit_id"]), u["unit_id"]
    assert u["slice"] in "ABC" and u["messages"][-1]["role"] in ("user", "tool"), u["unit_id"]
    if u["slice"] == "B":
        assert u["twin_of"] in byid and byid[u["twin_of"]]["group"] == u["group"], u["unit_id"]
    txt = json.dumps(u["messages"], ensure_ascii=False)
    assert not ban.search(txt), u["unit_id"]
    lens.append(sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", ""))) for m in u["messages"]))
print("units by slice:", dict(collections.Counter(u["slice"] for u in units)))
print("groups:", len({u["group"] for u in units}), " kinds:", dict(collections.Counter(u["kind"] for u in units)))
print(f"chars median {statistics.median(lens):.0f}, max {max(lens)}; over 24k: {sum(l > 24000 for l in lens)}")
step = max(1, len(units) // n_ex)
for u in units[::step][:n_ex]:
    print("\n--- example", u["unit_id"], u["slice"], u["kind"])
    for m in u["messages"]:
        c = m.get("content") or json.dumps([t["function"] for t in m.get("tool_calls", [])])
        print(f"[{m['role']}] {c[:300]!r}{' ...' if len(c) > 300 else ''}")
