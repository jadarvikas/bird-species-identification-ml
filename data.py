"""Loader for CUB-200-2011 (metadata, bounding boxes, class-level attributes)."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_cub(root):
    """Returns (df, class_names, class_attrs).

    df           : one row per image (id, path, y, is_train, x, y0, w, h) sorted by id
    class_names  : list of 200 species names
    class_attrs  : (200, 312) array, fraction of the class's images where each
                   attribute is present (the human-annotated 312 attributes the
                   reference report used - here only used as *training targets*,
                   never needed at prediction time).
    """
    root = Path(root)
    rd = lambda f, cols: pd.read_csv(root / f, sep=" ", header=None, names=cols)
    images = rd("images.txt", ["id", "path"])
    labels = rd("image_class_labels.txt", ["id", "cls"])
    split = rd("train_test_split.txt", ["id", "is_train"])
    bbox = pd.read_csv(root / "bounding_boxes.txt", sep=" ", header=None,
                       names=["id", "x", "y0", "w", "h"])
    classes = rd("classes.txt", ["cls", "name"])

    df = images.merge(labels).merge(split).merge(bbox).sort_values("id").reset_index(drop=True)
    df["y"] = df["cls"] - 1
    names = [n.split(".", 1)[-1].replace("_", " ") for n in classes["name"]]

    att = pd.read_csv(root / "attributes" / "image_attribute_labels.txt", sep=r"\s+",
                      header=None, usecols=[0, 1, 2], names=["id", "attr", "present"],
                      engine="python", on_bad_lines="skip")
    att = att.merge(df[["id", "y"]], on="id")
    mat = att.groupby(["y", "attr"])["present"].mean().unstack(fill_value=0.0)
    class_attrs = mat.reindex(range(len(names)), fill_value=0.0).to_numpy(dtype=np.float32)
    return df, names, class_attrs
