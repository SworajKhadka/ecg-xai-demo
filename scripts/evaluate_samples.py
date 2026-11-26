"""Run the stacked ensemble on every record in samples/samples.json.

Prints the cardiologist label vs the predicted classes and severity grade
for each record, then the number of exact label matches.

Usage:  python scripts/evaluate_samples.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ecgxai as E  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models")
SAMPLE_DIR = os.path.join(ROOT, "samples")


def predict(base, ens, meta):
    sig = E.fix_length(E.read_wfdb(base))
    xn = E.normalise(E.bandpass(sig),
                     np.array(meta["norm_mean"], np.float32),
                     np.array(meta["norm_std"], np.float32))
    P, S = ens.base_probs(xn[None])
    prob, grades, _, conf = ens.combine(P, S)
    prob = prob[0]
    positive = [c for k, c in enumerate(E.CLASSES) if prob[k] >= ens.thr[k]]
    if not positive:                                   # same fallback as the app
        positive = [E.CLASSES[int(np.argmax(prob))]]
    return positive, int(grades[0]), float(conf[0])


def main():
    models, meta = E.load_models(MODEL_DIR)
    ens = E.Ensemble(models, meta)
    with open(os.path.join(SAMPLE_DIR, "samples.json")) as f:
        info = json.load(f)

    header = f"{'record':<10} {'true':<16} {'predicted':<16} {'grade (true/pred)':<18} {'conf':>5}  match"
    print(header)
    print("-" * len(header))
    exact = 0
    for stem in sorted(info):
        truth = info[stem]
        pred, grade, conf = predict(os.path.join(SAMPLE_DIR, stem), ens, meta)
        ok = set(pred) == set(truth["labels"])
        exact += ok
        print(f"{stem:<10} {'/'.join(truth['labels']):<16} {'/'.join(pred):<16} "
              f"{str(truth['severity']) + ' / ' + str(grade):<18} {conf:>5.2f}  {'yes' if ok else 'no'}")
    print(f"\nExact label matches: {exact}/{len(info)}")


if __name__ == "__main__":
    main()
