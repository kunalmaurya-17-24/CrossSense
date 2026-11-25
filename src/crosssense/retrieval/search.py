"""Feature extraction for the non-trained statistical baseline."""

import numpy as np


def statistical_features(images: np.ndarray) -> np.ndarray:
    """Compute a weak eight-dimensional common descriptor from normalized patches.

    Average selected channels before mean/std/quantiles and edge statistics.
    Comparable dimensions do not imply physically aligned sensor statistics.
    This uses neither coordinates nor training labels.
    """
    array = np.asarray(images, dtype=np.float32)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Statistical baseline requires finite BCHW images")
    gray = array.mean(axis=1)
    flat = gray.reshape(len(gray), -1)
    features = np.stack([
        flat.mean(1), flat.std(1), *np.quantile(flat, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1),
        (np.abs(np.diff(gray, axis=1)).mean((1, 2)) +
         np.abs(np.diff(gray, axis=2)).mean((1, 2))) / 2,
    ], axis=1).astype(np.float32)
    # A constant zero patch still has a defined baseline descriptor.
    features[:, 0] += 1e-6
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
