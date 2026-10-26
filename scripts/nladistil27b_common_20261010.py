"""Shared pieces for the 27B NLA distillation (10 Oct 2026).

Teacher = ceselder/qwen3.6-27b-nla-rl (EasyNLA), the NLA WorkspaceBench scores: av_base + av_rl_adapters/iter_000400.
Conventions checked in EasyNLA@4d72847 and nanoNLA@3e3a62a:
- vector = output of decoder block 42 of Qwen3.6-27B (hook on layers[42]; HF hidden_states[43]) at the last token of
  the prefix, prefix fed alone (tokenizer add_special_tokens=True; Qwen adds no BOS), no normalisation.
- AV injection (Karvonen): at the output of block 1, marker position p (id 158983, neighbours 29 / 510):
  h_p <- h_p + |h_p| * v / |v|. Chat template with enable_thinking=False.
- AR: ar_reconstructor (43 blocks), lm_head and final norm removed, last real token of
  'Summary of the following text: <text>{explanation}</text> <summary>', normalised to sqrt(d), fp32 Linear(d, d) head.
- score: pred and gold both normalised to sqrt(d); mse = mean over elements = 2(1 - cos);
  fve = 1 - mse / rawvar, rawvar = MSE(v_norm, mean(v_norm)) over train vectors.
NLADISTIL_SMOKE=1 swaps in Qwen3.5-0.8B for subject, teacher and AR (random head) for a Mac pipeline test.
"""
import json, math, os, re
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SMOKE = os.environ.get('NLADISTIL_SMOKE') == '1'
TEACHER_REPO = 'ceselder/qwen3.6-27b-nla-rl'
AV_ADAPTER = 'av_rl_adapters/iter_000400'
SUBJECT = 'Qwen/Qwen3.6-27B'
LAYER, LAYERS_SAVED, D = 42, (28, 42, 52), 5120
AR_BLOCKS = 43
INJ_ID, LEFT_ID, RIGHT_ID, INJ_CHAR = 158983, 29, 510, '㈜'
AV_TEMPLATE = ('You are a meticulous AI researcher conducting an important investigation into activation vectors from a '
               'language model. Your overall task is to describe the semantic content of that activation vector.\n\n'
               'We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation '
               'for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets '
               'describing that vector.\n\nHere is the vector:\n\n<concept>{injection_char}</concept>\n\nPlease provide an explanation.')
AR_TEMPLATE = 'Summary of the following text: <text>{explanation}</text> <summary>'
TEXT_TEMPLATE = ('You are a meticulous AI researcher. Below is a text that a language model is reading. Describe the semantic '
                 'content of the model\'s internal state at the final token of the text, enclosed within <explanation> tags. '
                 'The explanation consists of 2-3 text snippets.\n\n<text>{text}</text>\n\nPlease provide an explanation.')
EXPL_RE = re.compile(r'<explanation>\s*(.*?)\s*(?:</explanation>|$)', re.DOTALL)
MAX_NEW = 256
if SMOKE:
    SUBJECT = 'Qwen/Qwen3.5-0.8B'
    LAYER, LAYERS_SAVED, D, AR_BLOCKS = 12, (6, 12, 18), 1024, 13
    MAX_NEW = 24
# 12 Oct: NLA_PROFILE=7b swaps the TEACHER side (subject model, layer, width, AR) for the released Qwen2.5-7B L20 NLA
# (kitft), keeping the Qwen3.5 student prompt above. Teacher-size comparison: both teachers -> the same Qwen3.5-2B.
PROFILE = os.environ.get('NLA_PROFILE', '27b')
if PROFILE == '7b':
    SUBJECT = 'Qwen/Qwen2.5-7B-Instruct'
    LAYER, LAYERS_SAVED, D, AR_BLOCKS = 20, (20,), 3584, 21
    AV7_REPO, AR7_REPO = 'kitft/nla-qwen2.5-7b-L20-av', 'kitft/nla-qwen2.5-7b-L20-ar'
MSE_SCALE = math.sqrt(D)


def teacher_dir():
    """Local snapshot of the teacher repo (av_base, the chosen adapter, ar_reconstructor)."""
    import os
    if os.environ.get('NLA_TEACHER_DIR'):  # box 2 (11 Oct): only ar_reconstructor + tokenizer files were downloaded
        return Path(os.environ['NLA_TEACHER_DIR'])
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(TEACHER_REPO, allow_patterns=['av_base/*', f'{AV_ADAPTER}/*', 'ar_reconstructor/*',
                                                                  'nla_meta.yaml', 'run_config.yaml']))


def check_meta(path):
    import yaml
    m = yaml.safe_load(Path(path).read_text())
    assert m['extraction']['layer_index'] == 42 and m['extraction']['d_model'] == 5120, m['extraction']
    assert m['tokens']['injection_token_id'] == INJ_ID, m['tokens']
    assert m['tokens']['injection_left_neighbor_id'] == LEFT_ID and m['tokens']['injection_right_neighbor_id'] == RIGHT_ID
    assert m['prompt_templates']['actor'] == AV_TEMPLATE, 'AV prompt drift'
    assert m['prompt_templates']['critic'] == AR_TEMPLATE, 'AR prompt drift'


def chat_ids(tok, content):
    ids = tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    if isinstance(ids, dict) or hasattr(ids, 'keys'):
        ids = ids['input_ids']
    return list(ids)


def av_prompt_ids(tok):
    """Chat-templated AV prompt ids and the marker position (checked against the sidecar neighbours)."""
    ids = chat_ids(tok, AV_TEMPLATE.format(injection_char=INJ_CHAR))
    p = [i for i, t in enumerate(ids) if t == INJ_ID]
    assert len(p) == 1 and ids[p[0] - 1] == LEFT_ID and ids[p[0] + 1] == RIGHT_ID, f'injection site drift {p}'
    return ids, p[0]


def text_prompt_ids(tok, text):
    return chat_ids(tok, TEXT_TEMPLATE.format(text=text))


def eos_ids(tok):
    return sorted({tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id} - {None})


def normalise(v, scale):
    return v / (v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) / scale).to(v.dtype)


def extract_expl(text):
    m = EXPL_RE.search(text)
    closed = '</explanation>' in text
    return (m.group(1).strip(), closed) if m else (text.strip(), False)


def rawvar_baseline(V):
    vn = normalise(V.float(), MSE_SCALE)
    return ((vn - vn.mean(0, keepdim=True)) ** 2).mean().item()


def decoder_layers(model):
    m = model
    for attr in ('base_model', 'model'):
        while hasattr(m, attr) and not hasattr(m, 'layers'):
            m = getattr(m, attr)
    if hasattr(m, 'language_model'):
        m = m.language_model
    return m.layers


def karvonen_hook(model, state):
    """HF path of the teacher's injection: state['v'] = [B, D] vectors, state['pos'] = marker position (same for all rows,
    prompts are identical). Applied only on the prompt forward (seq len > 1)."""
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        v = state.get('v')
        if v is None or h.shape[1] <= state['pos']:
            return out
        p = state['pos']
        hp = h[:, p, :]
        vu = v.to(h.device, h.dtype) / v.float().norm(dim=-1, keepdim=True).clamp_min(1e-9).to(h.device, h.dtype)
        h[:, p, :] = hp + hp.float().norm(dim=-1, keepdim=True).to(h.dtype) * vu
        return out
    return decoder_layers(model)[1].register_forward_hook(hook)


class AR:
    """Teacher's reconstructor. Smoke mode: Qwen3.5-0.8B cut to AR_BLOCKS with an identity head."""

    def __init__(self, dev, tdir=None, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from safetensors.torch import load_file
        self.norm_in = True
        if PROFILE == '7b':  # kitft AR: 21 blocks, final norm removed, bf16 head on the raw last hidden state
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(AR7_REPO))
            self.tok = AutoTokenizer.from_pretrained(path)
            m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
            assert m.config.num_hidden_layers == AR_BLOCKS and m.config.hidden_size == D, m.config
            m.model.norm = nn.Identity()
            self.body = m.model.to(dev).eval()
            self.head = nn.Linear(D, D, bias=False, dtype=torch.bfloat16)
            self.head.load_state_dict(load_file(str(path / 'value_head.safetensors')))
            self.head = self.head.to(dev).eval()
            self.dev, self.norm_in = dev, False
            return
        td = None if SMOKE else Path(tdir or teacher_dir())
        path = SUBJECT if SMOKE else td / 'ar_reconstructor'
        # ar_reconstructor ships no tokenizer files: use av_base's (same Qwen3.6 vocab); fail loudly if encoding is empty
        self.tok = AutoTokenizer.from_pretrained(SUBJECT if SMOKE else td / 'av_base')
        assert len(self.tok('Summary of the following text')['input_ids']) > 3, 'AR tokenizer broken'
        m, info = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, output_loading_info=True)
        bad = [k for k in info.get('missing_keys', []) if 'layers.' in k or 'embed' in k]
        assert not bad, f'AR weights missing: {bad[:5]} ({len(bad)})'  # only lm_head / final norm may be absent
        body = m.model
        if SMOKE:
            body.layers = body.layers[:AR_BLOCKS]
        assert len(body.layers) == AR_BLOCKS, len(body.layers)
        body.norm = nn.Identity()
        self.body = body.to(dev).eval()
        self.head = nn.Linear(D, D, bias=False, dtype=torch.float32)
        if SMOKE:
            nn.init.eye_(self.head.weight)
        else:
            sd = load_file(str(Path(path) / 'value_head.safetensors'))
            self.head.load_state_dict({'weight': next(iter(sd.values())).float()} if len(sd) == 1 else sd)
        self.head = self.head.to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def reconstruct(self, texts, bs=64, max_len=1024):
        prompts = [AR_TEMPLATE.format(explanation=t) for t in texts]
        enc = [self.tok(p, add_special_tokens=True, truncation=True, max_length=max_len)['input_ids'] for p in prompts]
        order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
        res = [None] * len(enc)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            L = max(len(enc[i]) for i in idx)
            ids = torch.zeros(len(idx), L, dtype=torch.long)
            att = torch.zeros(len(idx), L, dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, :len(enc[i])] = torch.tensor(enc[i]); att[j, :len(enc[i])] = 1
            h = self.body(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev), use_cache=False).last_hidden_state
            last = att.sum(1).to(self.dev) - 1
            hl = h[torch.arange(len(idx), device=self.dev), last]
            hl = normalise(hl.float(), MSE_SCALE) if self.norm_in else hl.to(self.head.weight.dtype)
            pred = self.head(hl).float().cpu()
            for j, i in enumerate(idx):
                res[i] = pred[j]
        return torch.stack(res)


def fve_stats(P, G, rawvar, groups=None, n_boot=1000, seed=0):
    """FVE of reconstructions P vs gold G, CI by bootstrap over groups (docs) if given, shuffled control."""
    import numpy as np
    c = torch.nn.functional.cosine_similarity(P.float(), G.float(), dim=-1)
    mse = (2 * (1 - c)).numpy()
    g = np.random.default_rng(seed)
    perm = g.permutation(len(P))
    cm = torch.nn.functional.cosine_similarity(P[perm].float(), G.float(), dim=-1)
    if groups is None:
        groups = list(range(len(P)))
    ug = sorted(set(groups)); gi = {u: [] for u in ug}
    for k, u in enumerate(groups):
        gi[u].append(k)
    boot = []
    for _ in range(n_boot):
        pick = g.choice(len(ug), len(ug))
        ix = np.concatenate([gi[ug[j]] for j in pick])
        boot.append(1 - mse[ix].mean() / rawvar)
    return {'fve': float(1 - mse.mean() / rawvar), 'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'fve_shuffled': float(1 - (2 * (1 - cm)).mean().item() / rawvar), 'cos_mean': float(c.mean()), 'n': len(P)}, c


class Adapter(nn.Module):
    """27B L42 vector -> k student input embeddings; input normalised to sqrt(d) then LayerNorm, scale learned."""

    def __init__(self, d_student, k=4, d_in=D):
        super().__init__()
        self.k, self.d = k, d_student
        self.ln = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, k * d_student)
        self.gain = nn.Parameter(torch.tensor(1.0))

    def init_to(self, emb_std):
        nn.init.normal_(self.proj.weight, std=emb_std / math.sqrt(self.proj.in_features))
        nn.init.zeros_(self.proj.bias)

    def forward(self, v):
        x = self.ln(normalise(v.float(), math.sqrt(v.shape[-1])))
        return (self.proj(x) * self.gain).view(-1, self.k, self.d)


def jl(path, rows):
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def rl(path):
    return [json.loads(x) for x in open(path)]


@torch.no_grad()
def sample_embeds(model, embeds, attn, eos, max_new=MAX_NEW, temperature=1.0):
    """Ancestral sampling from prompt embeddings [B, T, d] (left padded; mask attn). Pure temperature, no top-k / top-p.
    Returns token lists (EOS included if sampled)."""
    B, dev = embeds.shape[0], embeds.device
    eos_t = torch.tensor(eos, device=dev)
    out = model(inputs_embeds=embeds, attention_mask=attn, use_cache=True, logits_to_keep=1)
    pkv, logits = out.past_key_values, out.logits[:, -1].float()
    toks = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(max_new):
        nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1)[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, eos[0]), nxt)
        nc, dn = nxt.cpu().tolist(), done.cpu().tolist()
        for b in range(B):
            if not dn[b]:
                toks[b].append(nc[b])
        done = done | torch.isin(nxt, eos_t)
        if bool(done.all()):
            break
        attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device=dev)], 1)
        out = model(input_ids=nxt[:, None], attention_mask=attn, past_key_values=pkv, use_cache=True, logits_to_keep=1)
        pkv, logits = out.past_key_values, out.logits[:, -1].float()
    return toks


def left_pad_embeds(seqs, d, dtype, dev):
    T = max(s.shape[0] for s in seqs)
    E = torch.zeros(len(seqs), T, d, dtype=dtype, device=dev)
    M = torch.zeros(len(seqs), T, dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        E[i, T - s.shape[0]:] = s.to(dev, dtype); M[i, T - s.shape[0]:] = 1
    return E, M
