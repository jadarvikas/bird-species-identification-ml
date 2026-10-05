"""Train + evaluate bird species classifiers on CUB-200-2011 (classical ML + ANN only).

Usage:  python train.py --root /path/to/CUB_200_2011
"""
import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import load_cub
from features import build_raw, encode, fit_codebook
from models import Prep, blend, full_proba, greedy_ensemble, zoo


def metrics(P, y):
    top5 = np.argsort(-P, 1)[:, :5]
    return 100 * (P.argmax(1) == y).mean(), 100 * (top5 == y[:, None]).any(1).mean()


def fit_eval(Xtr, ytr, Xev, class_attrs, n_comp, n_classes, skip):
    prep = Prep(n_comp).fit(Xtr)
    Ztr, Zev = prep.transform(Xtr), prep.transform(Xev)
    models, probs = zoo(class_attrs, skip), {}
    for name, m in models.items():
        t = time.time()
        m.fit(Ztr, ytr)
        probs[name] = full_proba(m, Zev, n_classes)
        print(f"   fitted {name:<12} ({time.time() - t:5.0f}s)", flush=True)
    return prep, models, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to CUB_200_2011 folder")
    ap.add_argument("--crop", default="grabcut", choices=["bbox", "grabcut", "none"],
                    help="grabcut = automatic (matches prediction); bbox = ground-truth boxes")
    ap.add_argument("--k", type=int, default=300, help="visual words")
    ap.add_argument("--pca", type=int, default=512)
    ap.add_argument("--skip", default="", help="comma list of models to skip, e.g. RandomForest,KNN")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="model.joblib")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    skip = [s for s in a.skip.split(",") if s]

    df, names, class_attrs = load_cub(a.root)
    nc, y = len(names), df["y"].to_numpy()
    tr = df["is_train"].to_numpy() == 1
    print(f"{len(df)} images, {nc} species, train={tr.sum()} test={(~tr).sum()}")

    print("1/4 extracting features (cached after first run)...")
    H, D = build_raw(df, a.root, a.crop, a.cache, a.jobs)
    print("2/4 visual-word codebook...")
    km = fit_codebook(D[tr], a.k)
    X = np.hstack([H, encode(D, km)])
    print("    feature dim:", X.shape[1])

    print("3/4 validation stage (model selection + ensemble weights)...")
    idx = np.where(tr)[0]
    itr, iva = train_test_split(idx, test_size=0.2, stratify=y[idx], random_state=0)
    _, _, vprobs = fit_eval(X[itr], y[itr], X[iva], class_attrs, a.pca, nc, skip)
    weights = greedy_ensemble(vprobs, y[iva], nc)
    print("    validation acc:", {n: round(float(metrics(P, y[iva])[0]), 1) for n, P in vprobs.items()})
    print("    ensemble weights:", weights)

    print("4/4 final fit on full train set, single evaluation on official test set...")
    prep, models, tprobs = fit_eval(X[tr], y[tr], X[~tr], class_attrs, a.pca, nc, skip)
    tprobs["ENSEMBLE"] = blend(tprobs, weights)
    rows = [(n, *metrics(P, y[~tr])) for n, P in tprobs.items()]
    res = pd.DataFrame(rows, columns=["model", "top1_%", "top5_%"]).round(2).sort_values("top1_%")
    print("\n", res.to_string(index=False))
    res.to_csv("results.csv", index=False)

    joblib.dump(dict(codebook=km, prep=prep, weights=weights, names=names,
                     train_crop=a.crop,
                     models={n: models[n] for n in weights},
                     infer_crop=a.crop), a.out, compress=3)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
