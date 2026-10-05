"""Identify the bird in an image.   python predict.py photo.jpg [--model model.joblib]"""
import argparse

import cv2
import joblib
import numpy as np

import models  # noqa: F401  (needed to unpickle custom classes)
from features import encode, from_array


def load(path="model.joblib"):
    return joblib.load(path)


def predict_array(bundle, img_bgr, k=5, crop=None):
    if bundle.get("train_crop") == "bbox" and crop is None:
        raise ValueError(
            "This model was trained on ground-truth bounding boxes and cannot "
            "be used directly on new photos. Retrain with --crop grabcut or --crop none."
        )
    hand, d = from_array(img_bgr, None, crop or bundle["infer_crop"])
    x = np.hstack([hand, encode(d[None], bundle["codebook"])[0]])[None]
    Z = bundle["prep"].transform(x)
    w = bundle["weights"]
    P = sum(c * bundle["models"][n].predict_proba(Z) for n, c in w.items()) / sum(w.values())
    top = np.argsort(-P[0])[:k]
    return [(bundle["names"][i], float(P[0, i])) for i in top]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    for name, p in predict_array(load(a.model), img, a.k):
        print(f"{p * 100:5.1f}%  {name}")
