"""Default decisions at the 27B queue's gates (10 Oct 2026). Rules are fixed here, before results exist (plan s.13-15).
An override in LOG/decisions.json ({gate: value}) wins; Claude writes it from the Mac after reading results.
  vllm_check:  use vLLM if its notes parse >= 95% and its held-out FVE >= HF pre-flight FVE - 0.04 -> {"backend": ...}
  pilot:       10k pilot must beat shuffled by >= 0.2 FVE with >= 90% parsed, else exit 1
  blends_gate: run S3 if teacher notes on 50 blends parse >= 90% and their AR cos >= 0.8 x the natural held-out cos
  main_recipe: from the 66k probes take lr / passes that beat the base by >= 0.01 FVE; blends if the 131k blends arm
               beats the 131k base by >= 0.005 on natural held-out FVE. Last line printed = extra args for the 263k run.
"""
import argparse, json, sys
from pathlib import Path


def final_fve(run):
    p = Path(run) / 'done.json'
    if not p.exists():
        return None
    return json.load(open(p))['final'].get('heldout', {}).get('fve')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gate')
    ap.add_argument('--log', required=True)
    ap.add_argument('--tch', default=str(Path.home() / 'teacher27b'))
    ap.add_argument('--run', default='')
    ap.add_argument('--stem', default='')
    ap.add_argument('--dt', default='')
    a = ap.parse_args()
    L = Path(a.log)
    over = json.load(open(L / 'decisions.json')) if (L / 'decisions.json').exists() else {}
    out = {'gate': a.gate}
    if a.gate == 'vllm_check':
        pf = json.load(open(L / 'preflight' / 'preflight.json'))['teacher_hf_heldout50']
        v = json.load(open(Path(a.tch) / 'heldout' / 'T_000_ar.json'))
        ok = v['parsed'] >= 0.95 and v['fve'] >= pf['fve'] - 0.04
        out.update(hf_fve50=pf['fve'], vllm_fve=v['fve'], vllm_parsed=v['parsed'], backend='vllm' if ok else 'hf')
        if 'backend' in over:
            out['backend'] = over['backend']
    elif a.gate == 'pilot':
        d = json.load(open(Path(a.run) / 'done.json'))['final']['heldout']
        out.update(d)
        out['pass'] = d['fve'] - d['fve_shuffled'] >= 0.2 and d['parsed'] >= 0.9
        if 'pilot' in over:
            out['pass'] = bool(over['pilot'])
        print(json.dumps(out))
        sys.exit(0 if out['pass'] else 1)
    elif a.gate == 'blends_gate':
        pf = json.load(open(L / 'preflight' / 'preflight.json'))
        b, n = pf.get('blends50'), pf['teacher_hf_heldout50']
        out['run_s3'] = bool(b and b['parsed'] >= 0.9 and b['cos_mean'] >= 0.8 * n['cos_mean'])
        out.update(blends50=b, natural_cos=n['cos_mean'])
        if 'run_s3' in over:
            out['run_s3'] = bool(over['run_s3'])
    elif a.gate == 'main_recipe':
        r = lambda tag: final_fve(f'runs/{a.stem}_{tag}_{a.dt}')
        base66, base131 = r('pairs66k'), r('pairs131k')
        args, why = [], {}
        lrs = {'1e-5': base66, '5e-6': r('pairs66k-lr5e-6'), '2e-5': r('pairs66k-lr2e-5')}
        best = max((k for k in lrs if lrs[k] is not None), key=lambda k: lrs[k])
        if best != '1e-5' and base66 is not None and lrs[best] - base66 >= 0.01:
            args += ['--lr-lm', best]
        ep = {'1.5': base66, '1.0': r('pairs66k-ep1.0'), '2.0': r('pairs66k-ep2.0')}
        if ep['2.0'] is not None and base66 is not None and ep['2.0'] - base66 >= 0.01:
            args += ['--epochs', '2.0']
        bl = r('pairs131k-blends11')
        if bl is not None and base131 is not None and bl - base131 >= 0.005:
            args += ['--mix-blends', '0.11']
        why.update(lr=lrs, epochs=ep, blends={'base131': base131, 'blends11': bl})
        if 'main_args' in over:
            args = over['main_args'].split()
        out.update(why=why, args=args)
        print(json.dumps(out))
        print(' '.join(args))
        return
    print(json.dumps(out))


if __name__ == '__main__':
    main()
