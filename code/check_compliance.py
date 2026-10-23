"""Pre-submission compliance and audit checks.

Verifies the three things the guide measures beyond accuracy:

1. **Asset size** — total bytes of the checkpoint bundle (limit 16 MiB).
2. **Interface robustness** — ``predict()`` is called directly on a truncated,
   re-identified bank to confirm the method makes no assumption about the
   number of questions or the question IDs (the instructor's audit bank).
3. **Fresh-process shapes** — the returned choices are Python ``int`` (the
   scorer rejects ``numpy.int64``) and one-per-question in bank order.

Run from ``code/`` after training the bundle:

    python check_compliance.py --checkpoint runs/model_bundle
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import load_inputs, validate_choices


def load_method(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('student_method', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def bundle_size(path):
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', type=Path, default=Path('submission.py'))
    ap.add_argument('--checkpoint', type=Path, default=Path('runs/model_bundle'))
    ap.add_argument('--split', default='validation')
    ap.add_argument('--audit-n', type=int, default=37)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ok = True

    # 1. asset size ---------------------------------------------------------
    size = bundle_size(args.checkpoint)
    limit = 16 * 2 ** 20
    print(f'[1] bundle size      : {size:,} bytes ({size/2**20:.2f} MiB) '
          f'limit {limit:,}  -> {"OK" if size <= limit else "FAIL"}')
    ok &= size <= limit

    # 2. interface robustness ----------------------------------------------
    bank, arrays = load_inputs(args.split)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(bank), size=args.audit_n, replace=False)
    pick.sort()
    sub_bank = []
    for new_id, i in enumerate(pick):
        row = bank[i]
        sub_bank.append({'id': f'audit-{new_id:04d}', 'question': row['question'],
                         'candidates': row['candidates']})
    sub_arrays = {'embeddings': arrays['embeddings'][pick]}

    method = load_method(args.method)
    t0 = time.perf_counter()
    choices = method.predict(sub_bank, sub_arrays, args.checkpoint)
    dt = time.perf_counter() - t0

    types_ok = all(type(c) is int for c in choices)
    n_ok = len(choices) == len(sub_bank)
    validate_choices(sub_bank, [r['id'] for r in sub_bank], choices)
    print(f'[2] truncated bank   : n={len(sub_bank)} ids rewritten, '
          f'{len(choices)} choices, all python int={types_ok}, '
          f'order preserved={n_ok} -> {"OK" if types_ok and n_ok else "FAIL"}')
    ok &= types_ok and n_ok

    # order independence: shuffle and confirm the same rows get the same pick
    order = rng.permutation(len(sub_bank))
    sh_bank = [sub_bank[i] for i in order]
    sh_arrays = {'embeddings': sub_arrays['embeddings'][order]}
    sh_choices = method.predict(sh_bank, sh_arrays, args.checkpoint)
    match = all(sh_choices[j] == choices[order[j]] for j in range(len(order)))
    print(f'[3] order invariance : {"OK" if match else "FAIL"}')
    ok &= match

    # 3. full-split timing (predict only; scoring timed by score_predictions)
    tb = time.perf_counter()
    full_choices = method.predict(bank, arrays, args.checkpoint)
    ft = time.perf_counter() - tb
    print(f'[4] predict({args.split}, n={len(bank)}) : {ft:.2f}s')
    print(f'    predict(audit-n={args.audit_n})       : {dt:.3f}s')

    print('\nRESULT:', 'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
