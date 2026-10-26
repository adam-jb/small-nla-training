"""Shared pieces for the NLA distillation study (3 Oct 2026).

Teacher = released NLA for Qwen2.5-7B-Instruct layer 20 (kitft/nla-qwen2.5-7b-L20-av / -ar).
Details copied from kitft/natural_language_autoencoders@0577769 (nla_inference.py, nla/reward.py, nla/schema.py):
- vector = output of decoder block 20 (hook on model.model.layers[20]), raw text, position >= 50
- AV: vector rescaled to L2 norm 150 replaces the embedding of token 149705 in the chat-templated prompt
- AR: 21-layer truncated Qwen, final norm removed, Linear(d,d) head on the last token of
  'Summary of the following text: <text>{explanation}</text> <summary>' (add_special_tokens=True, no chat template)
- score: both vectors normalised to sqrt(d); mse = 2(1-cos); fve_nrm = 1 - mse / rawvar, rawvar = MSE(v_norm, mean)
"""
import re, json, math
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = 'Qwen/Qwen2.5-7B-Instruct'
AV_REPO = 'kitft/nla-qwen2.5-7b-L20-av'
AR_REPO = 'kitft/nla-qwen2.5-7b-L20-ar'
LAYER = 20
D = 3584
INJ_SCALE = 150.0
MSE_SCALE = 59.86651818838306
INJ_ID, LEFT_ID, RIGHT_ID = 149705, 29, 522
INJ_CHAR = '㈎'
AV_TEMPLATE = ('You are a meticulous AI researcher conducting an important investigation into activation vectors from a '
               'language model. Your overall task is to describe the semantic content of that activation vector.\n\n'
               'We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation '
               'for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets '
               'describing that vector.\n\nHere is the vector:\n\n<concept>{injection_char}</concept>\n\nPlease provide an explanation.')
AR_TEMPLATE = 'Summary of the following text: <text>{explanation}</text> <summary>'
TEXT_TEMPLATE = ('You are a meticulous AI researcher. Below is a text that a language model is reading. Describe the semantic '
                 'content of the model\'s internal state at the final token of the text, enclosed within <explanation> tags. '
                 'The explanation consists of 2-3 text snippets.\n\n<text>{text}</text>\n\nPlease provide an explanation.')
EXPL_RE = re.compile(r'<explanation>\s*(.*?)\s*</explanation>', re.DOTALL)
EOS_ID = 151645  # <|im_end|>

import os
SMOKE = os.environ.get('NLADISTIL_SMOKE') == '1'  # Mac pipeline test: Qwen2.5-0.5B stands in for subject and teacher
if SMOKE:
    SUBJECT = AV_REPO = 'Qwen/Qwen2.5-0.5B-Instruct'
    LAYER, D = 10, 896


def check_meta(meta_path):
    """Fail loudly if the downloaded sidecar disagrees with the constants above."""
    if SMOKE:
        return
    import yaml
    m = yaml.safe_load(Path(meta_path).read_text())
    assert m['d_model'] == D and m['extraction_layer_index'] == LAYER, m
    if m['role'] == 'av':
        assert m['extraction']['injection_scale'] == INJ_SCALE
        assert m['tokens']['injection_token_id'] == INJ_ID
        assert m['prompt_templates']['av'] == AV_TEMPLATE, 'AV prompt drift'
    else:
        assert abs(m['extraction']['mse_scale'] - MSE_SCALE) < 1e-6
        assert m['prompt_templates']['ar'] == AR_TEMPLATE, 'AR prompt drift'


def av_prompt_ids(tok):
    """Chat-templated AV prompt ids and the injection position (checked against the sidecar neighbours)."""
    content = AV_TEMPLATE.format(injection_char=INJ_CHAR)
    ids = tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True, add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, 'input_ids'):
        ids = ids['input_ids']
    ids = list(ids)
    p = [i for i, t in enumerate(ids) if t == INJ_ID]
    assert len(p) == 1 and ids[p[0] - 1] == LEFT_ID and ids[p[0] + 1] == RIGHT_ID, 'injection site drift'
    return ids, p[0]


def text_prompt_ids(tok, text):
    ids = tok.apply_chat_template([{'role': 'user', 'content': TEXT_TEMPLATE.format(text=text)}], tokenize=True,
                                  add_generation_prompt=True)
    if isinstance(ids, dict) or hasattr(ids, 'input_ids'):
        ids = ids['input_ids']
    return list(ids)


def normalise(v, scale):
    return v / (v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) / scale).to(v.dtype)


def extract_expl(text):
    m = EXPL_RE.search(text)
    return (m.group(1).strip(), True) if m else (text.replace('<explanation>', '').strip(), False)


def rawvar_baseline(V):
    """fve_nrm baseline: MSE(v_norm, mean) with v_norm at L2 = sqrt(d) (paper: ~0.72 for this layer)."""
    vn = normalise(V.float(), MSE_SCALE)
    return ((vn - vn.mean(0, keepdim=True)) ** 2).mean().item()


class Adapter(nn.Module):
    """7B layer-20 vector -> k student input embeddings. Input is normalised to unit L2 first, so scale is learned."""

    def __init__(self, d_student, k=4, d_in=D):
        super().__init__()
        self.k, self.d = k, d_student
        self.ln = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, k * d_student)
        self.gain = nn.Parameter(torch.tensor(1.0))

    def init_to(self, emb_std):
        """Start outputs at the size of ordinary token embeddings (LN output has unit variance per dim)."""
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
def sample(model, embeds, attn, max_new=200, temperature=1.0, topk=0, eos=EOS_ID):
    """Plain ancestral sampling from prompt embeddings (pure T, no top-k/top-p, as the AV's SGLang defaults).
    embeds [B,T,d] left padded, attn [B,T]. Returns token lists (EOS included if sampled) and, if topk>0, per-sequence
    (ids [n,topk] int32, logprobs [n,topk] fp16) of the T=1 distribution at each generated step."""
    B = embeds.shape[0]
    dev = embeds.device
    pos = (attn.cumsum(-1) - 1).clamp_min(0)
    out = model(inputs_embeds=embeds, attention_mask=attn, position_ids=pos, use_cache=True, logits_to_keep=1)
    pkv, logits = out.past_key_values, out.logits[:, -1].float()
    toks = [[] for _ in range(B)]
    tks = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    last_pos = pos[:, -1]
    for step in range(max_new):
        lp = torch.log_softmax(logits, -1)
        if topk:
            v, ix = lp.topk(topk, -1)
            v, ix = v.cpu(), ix.cpu()
        nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1)[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        nc = nxt.cpu().tolist()
        dn = done.cpu().tolist()
        for b in range(B):
            if not dn[b]:
                toks[b].append(nc[b])
                if topk:
                    tks[b].append((ix[b].int(), v[b].half()))
        done = done | (nxt == eos)
        if bool(done.all()):
            break
        attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device=dev)], 1)
        last_pos = last_pos + 1
        out = model(input_ids=nxt[:, None], attention_mask=attn, position_ids=last_pos[:, None], past_key_values=pkv,
                    use_cache=True, logits_to_keep=1)
        pkv, logits = out.past_key_values, out.logits[:, -1].float()
    if topk:
        tks = [(torch.stack([a for a, _ in t]).numpy(), torch.stack([b for _, b in t]).numpy()) if t else None for t in tks]
    return toks, tks


def left_pad_embeds(seqs, d, dtype, dev):
    """seqs: list of [T_i, d] tensors -> left-padded [B, T, d] and mask."""
    T = max(s.shape[0] for s in seqs)
    E = torch.zeros(len(seqs), T, d, dtype=dtype, device=dev)
    M = torch.zeros(len(seqs), T, dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        E[i, T - s.shape[0]:] = s.to(dev, dtype); M[i, T - s.shape[0]:] = 1
    return E, M
