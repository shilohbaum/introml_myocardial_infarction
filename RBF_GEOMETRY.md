# Geometric RBF classifier

The best-performing workflow combines handcrafted cardiac geometry and motion
summaries, a calibrated RBF SVM probability, and seven demographic features
in a CatBoost stack. The RBF probability is cross-fitted for each training
patient before CatBoost sees it, which avoids training the meta-model on
in-sample base-model scores. The autoencoder and broad multi-model comparison
scripts are not required.

Run the nested stacking evaluation from the repository root:

```bash
uv run python -m introml_myocardial_infarction.rbf_demographic_catboost \
  --output results/mi-rbf-demographics-catboost-stack-cv42
```

Its five-fold nested CV F1 is **0.909 ± 0.022** (seed 42). Each outer fold
selects the RBF settings, CatBoost configuration, and classification threshold
using only that fold's training partition. The full-training OOF score is a
selection score and is optimistic.

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

The standalone RBF refinement is a useful geometry-only comparator; its saved
run reports nested five-fold F1 of 0.881 ± 0.017. Test-set scores are omitted
because the cloud-to-CSV pairing is not independently verified.
