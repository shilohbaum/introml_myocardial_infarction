# Geometric RBF classifier

The RBF workflow uses handcrafted cardiac geometry and motion summaries,
followed by a calibrated RBF SVM. It does not use the autoencoder or the broad
multi-model comparison scripts.

Run the focused nested-CV refinement from the repository root:

```bash
uv run python -m introml_myocardial_infarction.rbf_refine_cv \
  --output results/mi-rbf-refine-cv42
```

`geometry_features.py` converts each point cloud to 702 features: per-surface,
per-frame centroid, spread, and covariance-eigenvalue summaries, plus
per-surface displacement-from-frame-zero and adjacent-frame distance
quantiles/statistics. The RBF pipeline imputes missing values, standardizes
features, calibrates SVM probabilities, and selects its decision threshold
inside training folds. `inter_surface_cv.py` contains an optional
order-independent nearest-surface-distance feature extraction experiment.

The saved refinement run reports nested five-fold F1 of 0.881 ± 0.017
(seed 42). This is the estimate for that 702-feature model and its candidate
search; other RBF feature variants in the broader experiment matrix scored
differently. The full-training OOF score in `cv_summary.json` is a selection
score and should not be treated as an independent estimate. Test-set scores
are omitted here because the cloud-to-CSV pairing is not independently
verified.
