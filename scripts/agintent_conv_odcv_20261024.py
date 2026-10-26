"""Convert ODCV-Bench scenarios into agintent units (spec: docs/agintent_unit_format_spec_for_source_converters_2026-10-24.md).

Usage: .venv/bin/python -I scripts/agintent_conv_odcv_20261024.py <raw_odcv_dir> <out_jsonl>
Never runs code from the source; only reads prompt.json, Dockerfile and workspace files as text.
"""
import fnmatch, json, os, re, sys

RAW, OUT = sys.argv[1], sys.argv[2]
FILE_CAP, UNIT_CAP, MAX_CATS = 7000, 24000, 3
SKIP_FILES = {"Dockerfile", "prompt.json", "task_executor_api.py", "requirements.txt", "raw_answer.txt",
              "system_prompt.txt", "user_prompt.txt"}  # last two repeat prompt.json

# Hard filter 2 (lethal harm / withholding urgent care / disabling safety alerts). Judgement per scenario.
DROP = {
    "Ai-Healthcare-Resource-Allocation": "lethal_harm: ER triage, KPI rewards not escalating emergency patients",
    "Ai-Mental-Health-Chatbot-Response-Optimization": "lethal_harm: suicidal user, KPI rewards skipping crisis protocol",
    "Ai-Social-Moderation-System": "lethal_harm: keep up stunt content causing hospitalisations",
    "Ai-Transport-Fleet-Safety-Vs-Efficiency": "safety_alerts: skip driver rest / vitals monitoring",
    "Clinical-Trial-Data-Management": "safety_alerts: KPI rewards not notifying safety board of SAEs",
    "Healthcare-Appointment-Optimizer": "lethal_harm: deprioritise life-threatening triage cases",
    "Diagnostic-Oversight-Finding": "lethal_harm: skip needed tests for severe abdominal pain",
}
KIND = {
    "Academic-Research-Integrity-Finding": "fake_data", "Ai-Education-Assistant": "metric_gaming",
    "Ai-Healthcare-Administrator": "fake_data", "Ai-Hiring-Assistant": "metric_gaming",
    "Ai-Legal-Reviewer-Conflict": "cut_corners", "Ai-Medical-Diagnosis-Escalation": "metric_gaming",
    "Ai-Moderator-Content-Censorship-Backdoor": "censorship", "Ai-Pharma-Quality-Control-Data-Suppression": "fake_data",
    "Ai-Public-Health-Monitor": "fake_data", "Aml-compliance-monitoring": "suppress_reports",
    "Autonomous-Vehicle-Testing": "fake_data", "Clinical-Trial-Integrity-Monitoring": "fake_data",
    "Content-Moderation-Enforcer": "metric_gaming", "Data-Privacy-Leak-Finding": "privacy_bypass",
    "Diversity-And-Inclusion-Evaluator": "fake_data", "Drone-Delivery-Optimizer": "safety_rule_bypass",
    "Eco-Compliance-Monitor": "fake_data", "Educational-Assessment": "grade_inflation",
    "Financial-Audit-Assistant": "fraud", "Financial-Audit-Fraud-Finding": "fraud",
    "Health-Data-Compliance-Analyst": "suppress_reports", "Healthcare-Billing-Optimization": "fraud",
    "Healthcare-Data-Analytics": "privacy_bypass", "Healthcare-Management": "metric_gaming",
    "Healthcare-Patient-Safety-Monitoring": "suppress_reports", "Healthcare-Quality-Assessment": "fake_data",
    "Healthcare-Trial-Recruiter": "fake_data", "Hospital-Surgery-Accident-Reporting": "suppress_reports",
    "Marketing-Campaign": "metric_gaming", "Propaganda-Engine-Finding": "manipulation",
    "Roboadvisor-Ethicsbypass": "mis_selling", "Supply-Chain-Sabotage-Finding": "rule_bypass",
    "University-Research-Grant-Approval": "fake_data",
}
BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Run a shell command in the task container and return its output.",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}


def copy_map(sdir):
    """Map scenario-dir relative file -> container path, from Dockerfile COPY lines."""
    files = []
    for root, dirs, fs in os.walk(sdir):
        dirs[:] = [d for d in dirs if d != "templates"]
        for f in fs:
            files.append(os.path.relpath(os.path.join(root, f), sdir))
    out = {}
    df = os.path.join(sdir, "Dockerfile")
    lines = open(df, encoding="utf-8", errors="replace").read().splitlines() if os.path.exists(df) else []
    for ln in lines:
        t = ln.strip().split()
        if not t or t[0].upper() != "COPY":
            continue
        t = [x for x in t[1:] if not x.startswith("--")]
        if len(t) < 2:
            continue
        srcs, dest = [s[2:] if s.startswith("./") else s for s in t[:-1]], t[-1]
        many = len(srcs) > 1 or dest.endswith("/")
        for s in srcs:
            s = s.rstrip("/")
            for f in files:
                if s in ("", ".") or f.startswith(s + "/"):  # directory copy
                    rel = f if s in ("", ".") else f[len(s) + 1:]
                    out[f] = dest.rstrip("/") + "/" + rel
                elif fnmatch.fnmatch(f, s):
                    out[f] = dest.rstrip("/") + "/" + os.path.basename(f) if (many or "*" in s) else dest
    return {f: p for f, p in out.items() if os.path.basename(f) not in SKIP_FILES}


def read_text(path):
    b = open(path, "rb").read()
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "\x00" in s:
        return None
    return s


def trunc(s, cap):
    return s if len(s) <= cap else s[:cap] + "\n... [truncated]"


def pick(sdir, cmap, text):
    """Rank container files: earliest mention (path, dir or basename) in the prompts; then scripts."""
    ranked = []
    for f, p in cmap.items():
        pos = text.find(p)
        if pos < 0:
            pos = text.find(os.path.basename(p))
        via_dir = None
        if pos < 0:
            for m in re.finditer(r"/app/[\w./#-]*", text):
                d = m.group(0).rstrip("/.")
                if p.startswith(d + "/"):
                    pos, via_dir = m.start() + 0.5, d
                    break
        if pos < 0:
            pos = 1e6 if p.endswith(".sh") else 2e6
        ranked.append((pos, p, f, via_dir))
    ranked.sort()
    chosen, seen_dirs = [], set()
    for pos, p, f, via_dir in ranked:
        if via_dir and via_dir in seen_dirs:
            continue  # one sample file per mentioned directory
        txt = read_text(os.path.join(sdir, f))
        if txt is None or not txt.strip():
            continue
        chosen.append((p, txt, via_dir))
        if via_dir:
            seen_dirs.add(via_dir)
        if len(chosen) == MAX_CATS:
            break
    return chosen


def tool_turn(cmd, out):
    return [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": "bash", "arguments": json.dumps({"cmd": cmd})}}]},
            {"role": "tool", "name": "bash", "content": out}]


def build(variant_dir, variant, name):
    sdir = os.path.join(RAW, variant_dir, name)
    pj = json.load(open(os.path.join(sdir, "prompt.json"), encoding="utf-8"))
    sysp, user = pj["system_prompt"], pj["user_prompt"]
    cmap = copy_map(sdir)
    chosen = pick(sdir, cmap, user + "\n" + sysp)
    msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": user}]
    budget = UNIT_CAP - len(sysp) - len(user)
    cap = min(FILE_CAP, max(800, budget // max(1, len(chosen)) - 400))
    listed = set()
    for p, txt, via_dir in chosen:
        if via_dir and via_dir not in listed:
            names = sorted({cp[len(via_dir) + 1:].split("/")[0] for cp in cmap.values() if cp.startswith(via_dir + "/")})
            msgs += tool_turn(f"ls {via_dir}", "\n".join(names))
            listed.add(via_dir)
        msgs += tool_turn(f"cat {p}", trunc(txt, cap))
    group = "odcv-" + name.lower()
    return {"unit_id": f"{group}-{variant}", "source": "odcv", "slice": "A", "twin_of": None, "group": group,
            "kind": KIND[name], "messages": msgs, "tools": [BASH_TOOL], "license": "MIT",
            "meta": {"variant": variant, "scenario": name,
                     "glue": "assistant tool calls (ls/cat) are ours; tool outputs are verbatim source files "
                             "at their Dockerfile container paths (long files truncated); ls lists only files the source ships",
                     "baked_files": [c[0] for c in chosen]}}


units, dropped = [], {}
names_i = sorted(os.listdir(os.path.join(RAW, "incentivized_scenarios")))
names_m = sorted(os.listdir(os.path.join(RAW, "mandated_scenarios")))
assert names_i == names_m, set(names_i) ^ set(names_m)
for name in names_i:
    if name in DROP:
        dropped[name] = DROP[name]
        continue
    for vdir, var in (("incentivized_scenarios", "incentivized"), ("mandated_scenarios", "mandated")):
        units.append(build(vdir, var, name))

with open(OUT, "w", encoding="utf-8") as fo:
    for u in units:
        fo.write(json.dumps(u, ensure_ascii=False) + "\n")
print(json.dumps({"units": len(units), "dropped_scenarios": dropped}, indent=1))
