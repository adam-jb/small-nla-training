"""Convert the Agent Misalignment Dataset 'shade_pressure' task prompts into agintent units.

Usage: .venv/bin/python -I scripts/agintent_conv_agent_misalignment_dataset_20261024.py <raw_repo_dir> <out_jsonl>

Licence: the full trajectory corpus (data/dataset/v1*.jsonl) is OpenRAIL-D and forbids training agents to
exhibit the documented boundary-crossing behaviours, so we DO NOT touch it. We use only the Tier-1 CC-BY-4.0
task descriptions: here, the static `user_query` and the shared honest system prompt from the
`src/shade_pressure/tasks/*.py` pressure tasks. We never run the source.

Scope choice: the dataset's three modes are shade_attack (explicit injected side task), shade_honest (control)
and petri. The injected side tasks are unauthorised money transfers and PII / API-key exfiltration to an
outsider -> spec hard filter 2 (leak to outsider / covert harm), so we drop the whole attack+honest-twin set
and keep only the `shade_pressure` autonomous-pressure tasks (no secondary task, honest prompt, synthetic data).
Those are single-mode, so every unit is slice A with no twin.
"""
import ast, json, os, re, sys

RAW, OUT = os.path.abspath(sys.argv[1]), sys.argv[2]
TDIR = os.path.join(RAW, "src", "shade_pressure", "tasks")
SKIP = {"base_pressure_task.py", "__init__.py"}


def fold(node):
    """Constant-fold a string expression (literal, implicit/explicit concat)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return fold(node.left) + fold(node.right)
    raise ValueError(f"non-constant user_query ({type(node).__name__})")


def prop_return(cls, name):
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            for s in ast.walk(n):
                if isinstance(s, ast.Return):
                    return fold(s.value)
    return None


# Honest system prompt is a module-level constant in base_pressure_task.py
base = ast.parse(open(os.path.join(TDIR, "base_pressure_task.py"), encoding="utf-8").read())
SYSTEM = None
for n in base.body:
    if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "HONEST_SYSTEM_PROMPT" for t in n.targets):
        # value is textwrap.dedent("...").strip()
        call = n.value
        while isinstance(call, ast.Call):
            if call.args and isinstance(call.args[0], ast.Constant):
                SYSTEM = call.args[0].value
                break
            call = call.func.value if isinstance(call.func, ast.Attribute) else None
assert SYSTEM, "could not extract honest system prompt"
import textwrap
SYSTEM = textwrap.dedent(SYSTEM).strip()


def kind_from_doc(doc):
    m = re.search(r"Misalignment:\s*(.+)", doc or "", re.S)
    desc = (m.group(1) if m else "").strip().replace("\n", " ")
    desc = re.sub(r"\s+", " ", desc)
    low = desc.lower()
    if "pay" in low or "payment" in low or "transaction" in low or "recipient" in low:
        kind = "overreach_payments"
    elif "share" in low or "forward" in low or "recipient" in low or "send" in low:
        kind = "over_sharing"
    elif "delete" in low or "remove" in low or "cleanup" in low or "offboard" in low:
        kind = "over_deletion"
    elif "sentiment" in low or "promotion" in low or "review" in low or "bias" in low:
        kind = "biased_judgement"
    else:
        kind = "scope_overreach"
    return kind, desc


def main():
    units, skipped = [], []
    for fn in sorted(os.listdir(TDIR)):
        if fn in SKIP or not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(TDIR, fn), encoding="utf-8").read())
        doc = ast.get_docstring(tree)
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)), None)
        if cls is None:
            continue
        try:
            uq = prop_return(cls, "user_query")
            env = prop_return(cls, "environment_type")
        except ValueError as e:
            skipped.append((fn, str(e)))
            continue
        if not uq:
            skipped.append((fn, "no user_query"))
            continue
        name = fn[:-3]
        kind, mis = kind_from_doc(doc)
        pressure = ""
        mp = re.search(r"Pressure:\s*(.+?)(?:\n\n|\nMisalignment:)", doc or "", re.S)
        if mp:
            pressure = re.sub(r"\s+", " ", mp.group(1)).strip()
        units.append({
            "unit_id": f"amd-{name.replace('_','-')}", "source": "agent_misalignment_dataset",
            "slice": "A", "twin_of": None, "group": f"amd-{name.replace('_','-')}", "kind": kind,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": uq}],
            "tools": [], "license": "CC-BY-4.0",
            "meta": {"task": name, "environment_type": env, "pressure": pressure, "misalignment": mis,
                     "mode": "shade_pressure (autonomous, no injected side task)",
                     "glue": "system prompt is the source's shared HONEST_SYSTEM_PROMPT; user is the task's "
                             "static user_query verbatim; the SHADE environment contents (external package) are "
                             "not included, so tools is empty"}})
    with open(OUT, "w", encoding="utf-8") as fo:
        for u in units:
            fo.write(json.dumps(u, ensure_ascii=False) + "\n")
    from collections import Counter
    print(json.dumps({"units": len(units), "skipped": skipped,
                      "kinds": dict(Counter(u["kind"] for u in units))}, indent=1))


main()
