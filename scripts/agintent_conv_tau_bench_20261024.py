"""Convert tau-bench (sierra-research/tau-bench, MIT) retail + airline tasks into agintent units. Task files and tool
files are Python: we read them with `ast` (literal values only) and never import or run them.

system = domain wiki.md (as tool_calling_agent does), tools = each tool's get_info() dict, user = the task's user
instruction turned into a first-person opening message (pronoun swap; user-simulator stage directions dropped).
slice A (kind "policy_violation"): the customer wants something the policy forbids (gold actions have no database
write, or only a transfer, or the instruction is pushy: insist/lie/pretend/...). Otherwise slice C.

Usage: .venv/bin/python -I scripts/agintent_conv_tau_bench_20261024.py <tau_bench_pkg_dir> <out_jsonl>
"""
import ast
import collections
import json
import os
import random
import re
import statistics
import sys

PKG, OUT = sys.argv[1], sys.argv[2]
CAP, MAX_CHARS = 400, 24000
TASK_FILES = {"retail": ["tasks_test.py", "tasks_dev.py", "tasks_train.py"], "airline": ["tasks_test.py"]}
READ = {"get_user_details", "get_order_details", "get_product_details", "find_user_id_by_email", "find_user_id_by_name_zip",
        "list_all_product_types", "calculate", "think", "get_reservation_details", "search_direct_flight",
        "search_onestop_flight", "list_all_airports"}
PUSHY = re.compile(r"\b(insist\w*|demand\w*|lie|lying|pretend\w*|claim that|claim to|falsely|not take no|don't take no|"
                   r"do not take no|against (the )?polic\w*|even though|even if (it|the agent)|try to convince|persuade|"
                   r"on behalf of|your friend's|another user|someone else's)\b", re.I)
# sentences that steer the user simulator rather than state what the customer wants
STAGE = re.compile(r"\b(agent's|rounds?|reactive|natural|make up|personality|conversation|you are (a )?(very |quite |a bit )?"
                   r"(polite|rude|emotional|impatient|patient|brief|firm|detail-oriented|cautious|logical|dependent|independent|"
                   r"organized|outgoing|curious|confident|flexible|insecure|shy|direct|busy|messy|sad|happy|optimistic|pessimistic|"
                   r"relaxing|creative|private|calm|angry|frustrated|upset|nervous|anxious|careful|friendly|lazy|smart|"
                   r"impulsive|polite but|in a hurry|not good)|do not reveal|don't reveal|not reveal|do not mention|don't mention|"
                   r"will not say|won't say|do not say|don't say|do not offer|do not provide|reveal)\b", re.I)

# imperative sentences that coach the simulator ("Be insistent ...", "Make sure to mention ...")
COACH = re.compile(r"^(Be|Make sure|Use|Try|Act|Pretend|Say|Tell|Mention|Insist|Remember|Keep|Do not|Don't|Never|Always|"
                   r"Only|Start|Begin|Then|Finally|First|At the end|In the end|Note|Express)\b")
ADJ_LIST = re.compile(r"^You are ([a-z-]+, )+(and )?[a-z-]+\.?$")
AGENT_VERB = {"is": "are", "has": "have", "does": "do", "tries": "try", "was": "were"}


def agent_to_you(m):
    v = m.group(2)
    if v in ("can", "cannot", "can't", "will", "won't", "should", "may", "might", "must", "would", "could", "to", "and", "or"):
        return "\x00 " + v
    v = AGENT_VERB.get(v, v[:-1] if v.endswith("s") and not v.endswith("ss") else v)
    return "\x00" + (" " + v if v else "")


def literal(n):
    if isinstance(n, ast.Call) and getattr(n.func, "id", None) in ("Task", "Action"):
        return {k.arg: literal(k.value) for k in n.keywords}
    if isinstance(n, ast.List):
        return [literal(e) for e in n.elts]
    if isinstance(n, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(n.keys, n.values)}
    return ast.literal_eval(n)


def load_tasks(path):
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign):
            return literal(node.value)


def load_tools(domain):
    tdir = os.path.join(PKG, "envs", domain, "tools")
    init = ast.parse(open(os.path.join(tdir, "__init__.py")).read())
    order = [e.id for n in init.body if isinstance(n, ast.Assign) for e in n.value.elts]
    by_class = {}
    for fn in os.listdir(tdir):
        if not fn.endswith(".py") or fn == "__init__.py":
            continue
        for cls in ast.parse(open(os.path.join(tdir, fn)).read()).body:
            if isinstance(cls, ast.ClassDef):
                for f in cls.body:
                    if isinstance(f, ast.FunctionDef) and f.name == "get_info":
                        by_class[cls.name] = ast.literal_eval(f.body[-1].value)
    return [by_class[c] for c in order]


SWAPS = [  # order matters
    (r"\byou are\b", "I am"), (r"\bYou are\b", "I am"), (r"\byou're\b", "I'm"), (r"\bYou're\b", "I'm"),
    (r"\byou were\b", "I was"), (r"\bYou were\b", "I was"), (r"\byou've\b", "I've"), (r"\bYou've\b", "I've"),
    (r"\byou'd\b", "I'd"), (r"\bYou'd\b", "I'd"), (r"\byou'll\b", "I'll"), (r"\bYou'll\b", "I'll"),
    (r"\byourself\b", "myself"), (r"\bYourself\b", "Myself"), (r"\byours\b", "mine"), (r"\byour\b", "my"), (r"\bYour\b", "My"),
    (r"\b(to|for|with|give|gives|tell|tells|ask|asks|help|helps|let|lets|send|sends|offer|offers|show|charge|charges|"
     r"call|email|from|about|refund|pay|contact|by|than|of) you\b", r"\1 me"),
    (r"\byou\b", "I"), (r"\bYou\b", "I"),
]


def first_person(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    keep, dropped = [], []
    for s in sents:
        (dropped if STAGE.search(s) or COACH.search(s) or ADJ_LIST.search(s) else keep).append(s)
    if not keep:  # never return an empty message
        keep, dropped = sents[:1], sents[1:]
    out = " ".join(keep)
    # "the agent says" -> "you say" (placeholder survives the you->I swap)
    out = re.sub(r"\b[Tt]he (?:customer service |airline |retail )?agent( (\w+))?", lambda m: agent_to_you(m) if m.group(2) else "\x00", out)
    for pat, rep in SWAPS:
        out = re.sub(pat, rep, out)
    out = out.replace("\x00", "you")
    out = re.sub(r"\bI name\b", "My name", out)
    out = re.sub(r"\band are willing\b", "and am willing", out)
    out = re.sub(r"\bI wants\b", "I want", out)
    out = re.sub(r"\bI is\b", "I am", out)
    out = re.sub(r"\bI has\b", "I have", out)
    out = re.sub(r"\bI does\b", "I do", out)
    return out, dropped


def main():
    drops, units, seen = collections.Counter(), [], set()
    for domain, files in TASK_FILES.items():
        wiki = open(os.path.join(PKG, "envs", domain, "wiki.md")).read()
        tools = load_tools(domain)
        for fn in files:
            for i, t in enumerate(load_tasks(os.path.join(PKG, "envs", domain, fn))):
                instr = t["instruction"]
                if instr in seen:
                    drops["duplicate_instruction"] += 1; continue
                seen.add(instr)
                if re.search(r"summitbridge|novastar", instr, re.I):
                    drops["summitbridge_novastar"] += 1; continue
                acts = [a["name"] for a in t["actions"]]
                writes = [a for a in acts if a not in READ]
                reasons = []
                if not writes:
                    reasons.append("no_write_in_gold")
                elif set(writes) <= {"transfer_to_human_agents"}:
                    reasons.append("transfer_only_in_gold")
                if PUSHY.search(instr):
                    reasons.append("pushy:" + PUSHY.search(instr).group(0).lower())
                msg, dropped = first_person(instr)
                messages = [{"role": "system", "content": wiki}, {"role": "user", "content": msg}]
                n = len(wiki) + len(msg) + len(json.dumps(tools))
                if n > MAX_CHARS:
                    drops["too_long"] += 1; continue
                uid = f"tau-{domain}-{fn.replace('tasks_', '').replace('.py', '')}-{i}"
                units.append({"unit_id": uid, "source": "tau_bench", "slice": "A" if reasons else "C", "twin_of": None,
                              "group": uid, "kind": "policy_violation" if reasons else "none", "messages": messages,
                              "tools": tools, "license": "MIT",
                              "meta": {"domain": domain, "task_file": fn, "task_index": i, "user_id": t.get("user_id"),
                                       "instruction_raw": instr, "gold_actions": acts, "a_reasons": reasons,
                                       "dropped_sentences": dropped, "chars": n,
                                       "glue": ["user turn = source instruction, 2nd->1st person pronoun swap (ours) and "
                                                "user-simulator stage-direction sentences dropped (listed in dropped_sentences)"]}})
    # all A; fill to CAP with C, airline C first (small domain), then retail C sampled with seed 0
    rng = random.Random(0)
    a = [u for u in units if u["slice"] == "A"]
    c_air = [u for u in units if u["slice"] == "C" and u["meta"]["domain"] == "airline"]
    c_ret = [u for u in units if u["slice"] == "C" and u["meta"]["domain"] == "retail"]
    rng.shuffle(c_ret)
    room = max(0, CAP - len(a) - len(c_air))
    out = a + c_air + c_ret[:room]
    drops["cap_C"] += len(c_ret) - min(room, len(c_ret))
    with open(OUT, "w") as f:
        for u in out:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    print("tasks kept", len(units), "written", len(out), "drops", dict(drops))
    for slc in "AC":
        us = [u for u in out if u["slice"] == slc]
        ch = [u["meta"]["chars"] for u in us]
        print(slc, len(us), "median chars", int(statistics.median(ch)), "max", max(ch),
              collections.Counter(u["meta"]["domain"] for u in us),
              collections.Counter(r.split(":")[0] for u in us for r in u["meta"]["a_reasons"]))


main()
