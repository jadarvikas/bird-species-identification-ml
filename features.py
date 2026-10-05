"""Hand-crafted (non-deep-learning) image features for bird identification.

Per image:
  * spatial HSV colour histograms (whole + 3x3 grid)   - plumage colours per body region
  * HOG                                                 - shape / pose / edges
  * uniform LBP histograms (whole + 2x2 grid)           - feather texture
  * Bag-of-Visual-Words over dense RootSIFT (k-means codebook, whole + 2x2 pyramid)
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import MiniBatchKMeans

IMG = 128        # working resolution of the bird crop
MAXSIDE = 256    # images are first shrunk to this (speed)
STEP = 8         # dense SIFT grid step
KP_XY = [(x, y) for y in range(STEP, IMG, STEP) for x in range(STEP, IMG, STEP)]
QUAD = np.array([(2 if y >= IMG / 2 else 0) + (1 if x >= IMG / 2 else 0) for x, y in KP_XY])


# ----------------------------------------------------------------- localisation
def grabcut_crop(img):
    """Classical foreground localisation (GrabCut) for images without a bbox."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * .06), int(h * .06), int(w * .88), int(h * .88))
    try:
        cv2.grabCut(img, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return img
    ys, xs = np.where((mask == 1) | (mask == 3))
    if len(ys) < 0.05 * h * w:
        return img
    m = 0.08
    y0, y1 = int(max(0, ys.min() - m * h)), int(min(h, ys.max() + m * h))
    x0, x1 = int(max(0, xs.min() - m * w)), int(min(w, xs.max() + m * w))
    return img[y0:y1, x0:x1] if (y1 - y0 > 16 and x1 - x0 > 16) else img


def _prep(img, bbox, crop):
    h, w = img.shape[:2]
    s = min(1.0, MAXSIDE / max(h, w))
    if s < 1:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    if crop == "bbox" and bbox is not None:
        x, y, bw, bh = [v * s for v in bbox]
        H, W = img.shape[:2]
        m = 0.1
        x0, x1 = int(max(0, x - m * bw)), int(min(W, x + bw * (1 + m)))
        y0, y1 = int(max(0, y - m * bh)), int(min(H, y + bh * (1 + m)))
        if x1 - x0 > 8 and y1 - y0 > 8:
            img = img[y0:y1, x0:x1]
    elif crop == "grabcut":
        img = grabcut_crop(img)
    return img


# ------------------------------------------------------------------- features
def _hsv_hist(region):
    parts = []
    for ch, bins, rng in ((0, 16, 180), (1, 8, 256), (2, 8, 256)):
        h = cv2.calcHist([region], [ch], None, [bins], [0, rng]).ravel()
        parts.append(np.sqrt(h / (h.sum() + 1e-9)))
    return np.concatenate(parts)


def _color(hsv):
    e = np.linspace(0, IMG, 4).astype(int)
    f = [_hsv_hist(hsv)]
    f += [_hsv_hist(hsv[e[i]:e[i + 1], e[j]:e[j + 1]]) for i in range(3) for j in range(3)]
    return np.concatenate(f)


def _lbp(gray):
    code = local_binary_pattern(gray, 16, 2, "uniform")
    hb = lambda c: np.sqrt(np.bincount(c.astype(int).ravel(), minlength=18) / c.size)
    m = IMG // 2
    return np.concatenate([hb(code)] + [hb(code[i:i + m, j:j + m]) for i in (0, m) for j in (0, m)])


def from_array(img, bbox=None, crop="grabcut"):
    """BGR uint8 image -> (hand-crafted vector, dense RootSIFT descriptors)."""
    img = _prep(img, bbox, crop)
    im = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    hsv, gray = cv2.cvtColor(im, cv2.COLOR_BGR2HSV), cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hand = np.concatenate([
        _color(hsv),
        hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)),
        _lbp(gray),
    ]).astype(np.float32)

    sift = cv2.SIFT_create()
    _, d = sift.compute(gray, [cv2.KeyPoint(float(x), float(y), 16) for x, y in KP_XY])
    if d is None or len(d) != len(KP_XY):
        d = np.zeros((len(KP_XY), 128), np.float32)
    d = np.sqrt(d / (d.sum(1, keepdims=True) + 1e-7))           # RootSIFT
    return hand, d.astype(np.float16)


def extract(path, bbox=None, crop="bbox"):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return from_array(img, bbox, crop)


def build_raw(df, root, crop="bbox", cache_dir="cache", n_jobs=-1):
    """Extract (and cache) features for every image in the dataframe."""
    root = Path(root).resolve()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    root_key = hashlib.sha256(str(root).casefold().encode("utf-8")).hexdigest()[:12]
    f = cache / f"raw_{crop}_{root_key}.npz"
    if f.exists():
        z = np.load(f)
        return z["H"], z["D"]
    jobs = (delayed(extract)(Path(root) / "images" / r.path, (r.x, r.y0, r.w, r.h), crop)
            for r in df.itertuples())
    res = Parallel(n_jobs=n_jobs, verbose=5)(jobs)
    H = np.stack([r[0] for r in res])
    D = np.stack([r[1] for r in res])
    np.savez(f, H=H, D=D)
    return H, D


# ---------------------------------------------------------------- bag of words
def fit_codebook(D, k=300, per_img=100, seed=0):
    rng = np.random.RandomState(seed)
    sample = np.concatenate([d[rng.choice(len(d), min(per_img, len(d)), replace=False)] for d in D])
    return MiniBatchKMeans(k, batch_size=4096, n_init=3, random_state=seed).fit(sample.astype(np.float32))


def encode(D, km):
    """(n, n_kp, 128) descriptors -> (n, 5K) Hellinger-normalised BoVW pyramid."""
    n, K = len(D), km.n_clusters
    words = km.predict(D.reshape(-1, D.shape[-1]).astype(np.float32)).reshape(n, -1)
    out = np.zeros((n, 5, K), np.float32)
    for i in range(n):
        out[i, 0] = np.bincount(words[i], minlength=K)
        for q in range(4):
            out[i, q + 1] = np.bincount(words[i][QUAD == q], minlength=K)
    out /= out.sum(2, keepdims=True) + 1e-9
    return np.sqrt(out).reshape(n, -1)
