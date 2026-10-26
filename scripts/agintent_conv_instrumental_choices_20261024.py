"""Convert Instrumental Choices (aisa-group) task x variant samples into agintent units.

Usage: .venv/bin/python -I scripts/agintent_conv_instrumental_choices_20261024.py <raw_repo_dir> <out_jsonl>

The source builds prompts and workspace files in Python. We never import or exec it. Instead a small
whitelisted AST walker below evaluates `_make_sample(variant_id, "benchmark")` using only pure string /
dict / list operations; file reads are limited to the repo's own asset dirs. Unknown calls raise.
"""
import ast, json, os, sys

RAW, OUT = os.path.abspath(sys.argv[1]), sys.argv[2]
TASKS_DIR = os.path.join(RAW, "inspect_ic", "tasks")
FILE_CAP, UNIT_CAP = 6000, 24000


class Unsupported(Exception):
    pass


class Ret(Exception):
    def __init__(self, v):
        self.v = v


class Module:
    cache = {}

    def __init__(self, path):
        self.path, self.dir = path, os.path.dirname(path)
        self.tree = ast.parse(open(path, encoding="utf-8").read())
        self.defs, self.imports, self.vals = {}, {}, {}
        for n in self.tree.body:
            if isinstance(n, ast.FunctionDef):
                self.defs[n.name] = n
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                tg = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in tg:
                    if isinstance(t, ast.Name):
                        self.defs[t.id] = n.value
            elif isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("inspect_ic."):
                p = os.path.join(RAW, *n.module.split(".")) + ".py"
                for a in n.names:
                    self.imports[a.asname or a.name] = (p, a.name)

    @classmethod
    def load(cls, path):
        if path not in cls.cache:
            cls.cache[path] = Module(path)
        return cls.cache[path]

    def lookup(self, name):
        if name in self.vals:
            return self.vals[name]
        if name in self.defs:
            d = self.defs[name]
            v = ("fn", self, d) if isinstance(d, ast.FunctionDef) else Interp(self).ev(d, {})
            self.vals[name] = v
            return v
        if name in self.imports:
            p, n = self.imports[name]
            return Module.load(p).lookup(n)
        raise Unsupported(f"name {name} in {os.path.basename(self.path)}")


def read_asset(base, rel):
    p = os.path.realpath(os.path.join(base, "assets", rel))
    if not p.startswith(os.path.realpath(RAW) + os.sep):
        raise Unsupported("asset path escapes repo")
    return open(p, encoding="utf-8").read()


def record(kind):
    return lambda *a, **k: dict(k, _kind=kind)


SAFE_BUILTINS = {"str": str, "int": int, "len": len, "range": range, "list": list, "dict": dict, "tuple": tuple,
                 "sorted": sorted, "set": set, "bool": bool, "min": min, "max": max, "any": any, "all": all,
                 "None": None, "True": True, "False": False}
CTORS = {"Sample": record("Sample"), "ChatMessageSystem": record("system"), "ChatMessageUser": record("user")}
STR_METHODS = {"replace", "strip", "rstrip", "lstrip", "lower", "upper", "split", "splitlines", "startswith",
               "endswith", "format", "join"}
DICT_METHODS = {"get", "items", "keys", "values", "update", "copy"}


class Interp:
    def __init__(self, mod):
        self.mod, self.steps = mod, 0

    def name(self, n, env):
        if n in env:
            return env[n]
        if n in ("_asset_text",):
            return ("py", lambda rel: read_asset(self.mod.dir, rel))
        if n == "shared_asset_text":
            return ("py", lambda rel: read_asset(os.path.join(TASKS_DIR, "_base"), rel))
        if n in CTORS:
            return ("py", CTORS[n])
        if n in SAFE_BUILTINS:
            return ("py", SAFE_BUILTINS[n]) if callable(SAFE_BUILTINS[n]) else SAFE_BUILTINS[n]
        return self.mod.lookup(n)

    def call(self, f, args, kw):
        if isinstance(f, tuple) and f[0] == "py":
            return f[1](*args, **kw)
        if isinstance(f, tuple) and f[0] == "fn":
            _, mod, fd = f
            a = fd.args
            params = [p.arg for p in a.posonlyargs + a.args]
            env = {}
            defaults = a.defaults
            for p, d in zip(params[len(params) - len(defaults):], defaults):
                env[p] = Interp(mod).ev(d, {})
            for p, d in zip([p.arg for p in a.kwonlyargs], a.kw_defaults):
                if d is not None:
                    env[p] = Interp(mod).ev(d, {})
            env.update(zip(params, args))
            env.update(kw)
            try:
                Interp(mod).block(fd.body, env)
            except Ret as r:
                return r.v
            return None
        raise Unsupported(f"call {f!r}")

    def ev(self, n, env):
        self.steps += 1
        if self.steps > 200000:
            raise Unsupported("step limit")
        t = type(n)
        if t is ast.Constant:
            return n.value
        if t is ast.Name:
            return self.name(n.id, env)
        if t is ast.JoinedStr:
            out = []
            for v in n.values:
                if isinstance(v, ast.Constant):
                    out.append(v.value)
                else:
                    x = self.ev(v.value, env)
                    if v.conversion == ord("r"):
                        x = repr(x)
                    spec = self.ev(v.format_spec, env) if v.format_spec else ""
                    out.append(format(x, spec))
            return "".join(out)
        if t is ast.BinOp:
            l, r = self.ev(n.left, env), self.ev(n.right, env)
            if isinstance(n.op, ast.Add):
                return l + r
            if isinstance(n.op, ast.Sub):
                return l - r
            if isinstance(n.op, ast.Mult):
                return l * r
            raise Unsupported("binop")
        if t is ast.UnaryOp:
            v = self.ev(n.operand, env)
            if isinstance(n.op, ast.Not):
                return not v
            if isinstance(n.op, ast.USub):
                return -v
            raise Unsupported("unary")
        if t is ast.BoolOp:
            if isinstance(n.op, ast.And):
                v = True
                for x in n.values:
                    v = self.ev(x, env)
                    if not v:
                        return v
                return v
            v = False
            for x in n.values:
                v = self.ev(x, env)
                if v:
                    return v
            return v
        if t is ast.Compare:
            l = self.ev(n.left, env)
            for op, c in zip(n.ops, n.comparators):
                r = self.ev(c, env)
                ok = {ast.Eq: lambda: l == r, ast.NotEq: lambda: l != r, ast.In: lambda: l in r,
                      ast.NotIn: lambda: l not in r, ast.Is: lambda: l is r, ast.IsNot: lambda: l is not r,
                      ast.Lt: lambda: l < r, ast.Gt: lambda: l > r, ast.LtE: lambda: l <= r,
                      ast.GtE: lambda: l >= r}[type(op)]()
                if not ok:
                    return False
                l = r
            return True
        if t is ast.IfExp:
            return self.ev(n.body, env) if self.ev(n.test, env) else self.ev(n.orelse, env)
        if t in (ast.List, ast.Tuple, ast.Set):
            vals = []
            for e in n.elts:
                if isinstance(e, ast.Starred):
                    vals.extend(self.ev(e.value, env))
                else:
                    vals.append(self.ev(e, env))
            return tuple(vals) if t is ast.Tuple else (set(vals) if t is ast.Set else vals)
        if t is ast.Dict:
            d = {}
            for k, v in zip(n.keys, n.values):
                if k is None:
                    d.update(self.ev(v, env))
                    continue
                key = self.ev(k, env)
                try:
                    d[key] = self.ev(v, env)
                except (Unsupported, TypeError, KeyError) as e:  # keep going: we only need a few files
                    d[key] = Unsupported(f"{type(e).__name__}: {e}")
            return d
        if t is ast.Subscript:
            return self.ev(n.value, env)[self.ev(n.slice, env)]
        if t in (ast.ListComp, ast.GeneratorExp):
            if len(n.generators) != 1:
                raise Unsupported("comp")
            g, out = n.generators[0], []
            for item in self.ev(g.iter, env):
                e2 = dict(env)
                self.assign(g.target, item, e2)
                if all(self.ev(c, e2) for c in g.ifs):
                    out.append(self.ev(n.elt, e2))
            return out
        if t is ast.Attribute:
            base = n.value
            if isinstance(base, ast.Name) and base.id in ("os", "json"):
                return ("mod", base.id, n.attr)
            if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id == "os":
                return ("mod", "os." + base.attr, n.attr)
            raise Unsupported(f"attr {n.attr}")
        if t is ast.Call:
            args = []
            for a in n.args:
                if isinstance(a, ast.Starred):
                    args.extend(self.ev(a.value, env))
                else:
                    args.append(self.ev(a, env))
            kw = {}
            for k in n.keywords:
                if k.arg is None:
                    kw.update(self.ev(k.value, env))
                else:
                    kw[k.arg] = self.ev(k.value, env)
            fn = n.func
            if isinstance(fn, ast.Attribute):
                if isinstance(fn.value, ast.Name) and fn.value.id in ("os", "json") or (
                        isinstance(fn.value, ast.Attribute) and isinstance(fn.value.value, ast.Name)
                        and fn.value.value.id == "os"):
                    full = ast.unparse(fn)
                    if full in ("os.getenv", "os.environ.get"):
                        return args[1] if len(args) > 1 else kw.get("default")
                    if full == "json.dumps":
                        try:
                            return json.dumps(*args, **kw)
                        except TypeError as e:
                            raise Unsupported(f"json.dumps: {e}")
                    if full == "json.loads":
                        return json.loads(*args)
                    raise Unsupported(full)
                obj = self.ev(fn.value, env)
                if isinstance(obj, str) and fn.attr in STR_METHODS:
                    return getattr(obj, fn.attr)(*args, **kw)
                if isinstance(obj, dict) and fn.attr in DICT_METHODS:
                    return getattr(obj, fn.attr)(*args, **kw)
                if isinstance(obj, list) and fn.attr in ("append", "extend", "copy"):
                    return getattr(obj, fn.attr)(*args, **kw)
                raise Unsupported(f"method {fn.attr} on {type(obj).__name__}")
            return self.call(self.ev(fn, env), args, kw)
        raise Unsupported(f"expr {t.__name__}")

    def assign(self, tg, v, env):
        if isinstance(tg, ast.Name):
            env[tg.id] = v
        elif isinstance(tg, (ast.Tuple, ast.List)):
            for a, b in zip(tg.elts, v):
                self.assign(a, b, env)
        elif isinstance(tg, ast.Subscript):
            self.ev(tg.value, env)[self.ev(tg.slice, env)] = v
        else:
            raise Unsupported("assign target")

    def block(self, body, env):
        for s in body:
            t = type(s)
            if t is ast.Return:
                raise Ret(self.ev(s.value, env) if s.value else None)
            if t is ast.Assign:
                v = self.ev(s.value, env)
                for tg in s.targets:
                    self.assign(tg, v, env)
            elif t is ast.AnnAssign:
                if s.value is not None:
                    self.assign(s.target, self.ev(s.value, env), env)
            elif t is ast.AugAssign:
                cur = self.ev(s.target, env)
                if not isinstance(s.op, ast.Add):
                    raise Unsupported("augassign")
                self.assign(s.target, cur + self.ev(s.value, env), env)
            elif t is ast.If:
                self.block(s.body if self.ev(s.test, env) else s.orelse, env)
            elif t is ast.For:
                for item in self.ev(s.iter, env):
                    self.assign(s.target, item, env)
                    self.block(s.body, env)
            elif t is ast.Expr:
                if not isinstance(s.value, ast.Constant):
                    self.ev(s.value, env)
            elif t is ast.Pass:
                pass
            elif t is ast.Raise:
                raise Unsupported("raise in source")
            else:
                raise Unsupported(f"stmt {t.__name__}")


# ---- per-task choices (ours) --------------------------------------------------------------
# Which workspace files to pre-bake (1-3, the docs that carry the official path and the shortcut hint),
# and a short tag for the instrumental shortcut each task tempts (from each task README "IC shortcut").
TASKS = {
    "quota_boost": ("account_misuse", ["{WORK_DIR}/README.md", "/docs/migration_notes.md", "/docs/coworker_note.md"]),
    "budget_siphon": ("resource_grab", ["{WORK_DIR}/README.md", "/ops/policies/budget_policy.md",
                                         "/ops/notes/budget_incident_notes.txt"]),
    "deployment_guardian": ("disable_oversight_control", ["/docs/runbook.md", "/docs/maintenance_policy.md",
                                                          "/docs/team_notes.md"]),
    "final_instance": ("self_preservation", ["{WORK_DIR}/README.md", "/docs/vm_policy.md", "/etc/vm/lifecycle.conf"]),
    "ledger_audit": ("tamper_records", ["{WORK_DIR}/draft_report.md", "/docs/runbook.md", "/docs/coworker_note.md"]),
    "memory_keeper": ("unauthorised_persistence", ["/docs/support_handbook.md", "/docs/shift_notes.md",
                                                   "/docs/coworker_note.md"]),
    "training_preserver": ("disable_shutdown_control", ["/docs/cost_policy.md", "/training/logs/cost_warnings.txt",
                                                        "/docs/coworker_note.md"]),
}
NO_PRESSURE = "no_dilemma_permitted_unwatched"
TWIN_PARTNER = "dilemma_permitted_unwatched"  # same clarity/observation/importance, only the dilemma differs
BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Run a shell command in the sandboxed workspace and return its output.",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}


def tool_turn(cmd, out):
    return [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": "bash", "arguments": json.dumps({"cmd": cmd})}}]},
            {"role": "tool", "name": "bash", "content": out}]


def trunc(s, cap):
    return s if len(s) <= cap else s[:cap] + "\n... [truncated]"


def main():
    units = []
    vp = Module.load(os.path.join(TASKS_DIR, "_base", "variant_profiles.py"))
    variant_ids = vp.lookup("BENCHMARK_VARIANT_IDS")
    for task, (kind, bake) in TASKS.items():
        mod = Module.load(os.path.join(TASKS_DIR, task, f"task_{task}.py"))
        work_dir = mod.lookup("WORK_DIR")
        for vid in variant_ids:
            s = Interp(mod).call(mod.lookup("_make_sample"), [vid, "benchmark"], {})
            sysm, userm = s["input"]
            files = s["files"]
            msgs = [{"role": "system", "content": sysm["content"]}, {"role": "user", "content": userm["content"]}]
            paths = [p.replace("{WORK_DIR}", work_dir) for p in bake]
            listing = sorted(os.path.relpath(p, work_dir) for p in files if p.startswith(work_dir + "/"))
            msgs += tool_turn(f"ls {work_dir}", "\n".join(listing))
            budget = UNIT_CAP - sum(len(m["content"]) for m in msgs)
            cap = min(FILE_CAP, budget // len(paths) - 300)
            for p in paths:
                v = files[p]
                if isinstance(v, Exception):
                    raise SystemExit(f"{task}/{vid}: could not render {p}: {v}")
                msgs += tool_turn(f"cat {p}", trunc(v, cap))
            slice_ = "B" if vid == NO_PRESSURE else "A"
            meta = {k: s["metadata"].get(k) for k in ("variant", "variant_id", "ic_usefulness", "instruction_stance",
                                                      "monitoring_level", "stakes_level", "honest_path_access",
                                                      "existential")}
            meta.update(task=task, sample_id=s["id"], baked_files=paths,
                        glue="assistant ls/cat tool calls are ours; ls lists the source's work-dir files; "
                             "cat outputs are the source's per-variant file text (rendered from its Python without running it)")
            units.append({"unit_id": f"ic-{task}-{vid}".replace("_", "-"), "source": "instrumental_choices",
                          "slice": slice_, "twin_of": f"ic-{task}-{TWIN_PARTNER}".replace("_", "-") if slice_ == "B" else None,
                          "group": f"ic-{task}".replace("_", "-"), "kind": kind, "messages": msgs,
                          "tools": [BASH_TOOL], "license": "MIT", "meta": meta})
    with open(OUT, "w", encoding="utf-8") as fo:
        for u in units:
            fo.write(json.dumps(u, ensure_ascii=False) + "\n")
    print(json.dumps({"units": len(units)}))


main()
