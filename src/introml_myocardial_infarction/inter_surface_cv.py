"""Nested-CV test of phase-specific inter-surface distances."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .dataloader import CardiacDataLoader
from .geometry_features import feature_matrix, labels_for, nested_evaluate


def inter_surface(cloud, points_per_surface=512, seed=31415):
    """Describe wall thickness and separation between each surface pair.

    Distances are bidirectional nearest-neighbour measurements within each
    phase, so they are invariant to point order, rigid translation, and
    rotation. Independent frame samples avoid index correspondence assumptions.
    """
    xyz = np.asarray(cloud[..., :3], dtype=np.float64)
    if xyz.ndim != 3 or not np.isfinite(xyz).all():
        raise ValueError("Expected finite cardiac point-cloud XYZ")
    codes = cloud[0, :, 3] if cloud.shape[-1] == 4 else np.zeros(xyz.shape[1])
    if cloud.shape[-1] == 4 and not np.all(cloud[..., 3] == codes[None, :]):
        raise ValueError("Surface codes must be stable across frames")
    surfaces = np.unique(codes)
    if len(surfaces) != 3:
        raise ValueError(f"Expected three surfaces, found {len(surfaces)}")
    rng = np.random.default_rng(seed)
    output = []
    for frame in xyz:
        clouds = [frame[codes == code] for code in surfaces]
        trees = [cKDTree(points) for points in clouds]
        for a in range(3):
            for b in range(a + 1, 3):
                pa, pb = clouds[a], clouds[b]
                na, nb = min(points_per_surface, len(pa)), min(points_per_surface, len(pb))
                qa = pa[rng.choice(len(pa), na, replace=False)]
                qb = pb[rng.choice(len(pb), nb, replace=False)]
                distances = np.r_[trees[b].query(qa, workers=1)[0],
                                  trees[a].query(qb, workers=1)[0]]
                output.extend(np.quantile(distances, [0.1, 0.25, 0.5, 0.75, 0.9]))
    result = np.asarray(output, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Inter-surface distance features are nonfinite")
    return result


def extract(loader):
    rows = []
    for i, sid in enumerate(loader.demographics.index):
        rows.append(inter_surface(loader.load_point_cloud(sid, mmap_mode="r"), seed=31415+i))
        if (i + 1) % 100 == 0 or i + 1 == len(loader):
            print(f"Extracted inter-surface distances for {i+1}/{len(loader)} patients", flush=True)
    matrix = np.stack(rows)
    return pd.DataFrame(matrix, columns=[f"pairdist_{i:03d}" for i in range(matrix.shape[1])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/training_data")
    parser.add_argument("--samples-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty")
    loader = CardiacDataLoader(args.data_dir, samples_dir=args.samples_dir)
    labels = labels_for(loader)
    pair_features = extract(loader)
    geometry = feature_matrix(loader)
    features = pd.concat([geometry.add_prefix("global_"), pair_features], axis=1)
    folds, probabilities = nested_evaluate(features, labels, n_splits=5, seed=42)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_csv(output / "nested_cv_folds.csv", index=False)
    pd.DataFrame(dict(sample_id=loader.demographics.index, label=labels,
                      probability=probabilities)).to_csv(output / "nested_oof_predictions.csv", index=False)
    pair_features.insert(0, "sample_id", loader.demographics.index.to_numpy())
    pair_features.to_csv(output / "inter_surface_features.csv", index=False)
    summary = dict(representation="global geometric summaries plus phase-specific inter-surface distances",
                   n_samples=len(labels), n_features=features.shape[1], n_splits=5, seed=42,
                   nested_mean_f1=float(np.mean([row["f1"] for row in folds])),
                   nested_sd_f1=float(np.std([row["f1"] for row in folds], ddof=1)),
                   nested_mean_auroc=float(np.mean([row["auroc"] for row in folds])))
    (output / "cv_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
