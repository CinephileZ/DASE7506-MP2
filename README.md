# DASE7506-MP2: Selecting Reliable LLM Reasoning
 
**Test Accuracy:** **84.37%** (394/467)

---

## Task Summary

Given a GSM8K math question and four candidate solutions from Qwen3-0.6B (with fixed 1024-d embeddings), select the candidate with the correct final numeric answer. The scorer checks only the final answer (including equivalent fractions/decimals), not reasoning validity. Multiple candidates can be correct. And any correct choice earns the point.

**Data:** 4,755 train / 798 val / 467 test.

**Constraints:** CPU inference ≤30s (4 threads), ≤16 MiB assets, seed 7506, no external data/lookups, must generalise to unseen audit bank.

---

## Method (Frozen)

**Model:** Linear group-softmax CE  
**Training:** Hard subset of train (n=2,114, questions with ≥2 distinct answers)  
**Optimiser:** L-BFGS two-start (deterministic)  
**Regularisation:** Per-feature L2 — hand 0.001, embedding D 0.4  
**Selection:** 10-fold grouped OOF on train-hard, confirmed on val-hard (bootstrap CI)

---

## Results

**Test Accuracy:** **84.37%** (394/467)

---

## Reproduction

### Quick Test (No Retraining)

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_full.txt
python check_data.py

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint final_checkpoint/model_bundle_v2 \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```


### Full Retraining

```bash
# Train (reproduces exactly)
python train.py --output runs/model_bundle_repro --folds 10 --seed 7506

# Test
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint runs/model_bundle_repro \
  --split test --output runs/test-predictions_repro.json
python score_predictions.py --split test \
  --predictions runs/test-predictions_repro.json --output runs/test-score_repro.json
```

---

## Compliance & Resources

| Resource | Used | Limit |
|----------|------|-------|
| Inference assets | 48,570 B | 16 MiB |
| CPU prediction (test) | 0.43 s | 30 s |
| Threads | 4 | 4 |
| Training data | Train-hard only | Train only |
| External data/lookup | None | Prohibited |

---

## AI Use Disclosure

Code authored by student. Claude Code used for: refactoring `mp2lib/`, drafting training script, writing verification parser, formatting report. All design decisions (feature groups, hard-subset training, per-block L2, group-softmax CE) made by student.

---

## License

Questions/references from GSM8K (MIT). Candidate solutions/embeddings from Qwen3-0.6B. See `code/data/LICENSE.txt` and `code/data/NOTICE.txt`.