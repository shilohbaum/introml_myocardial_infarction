## Dataset ##

# About the data

The dataset contains cardiac anatomies represented as point clouds in the folder `mixed_samples`.

These point clouds are derived using the deep learning pipeline by Beetz et al. [1]. Each MR image is segmented and contours are stacked to form a sparse point cloud. A geometric deep learning model then aligns contours and produces a dense reconstruction of overall shape. Figure 1 outlines this procedure.

Figure 1- Point cloud reconstruction method presented by Beetz et al.

Typical cine MRI has 25 frames per cardiac cycle. UK Biobank has 50 frames. We provide 10 frames per sample to show the heart at 10 points in the cardiac cycle. Each 3D image has a resolution of $1.8\times1.8\times8$ mm.

Shape: [10, 18000, 4]
the fourth column in the final dimension corresponds to which heart surface it belongs to: LV endocardium, LV epicardium, RV endocardium

The point clouds also have associated demographic characteristics in `mixed_demographics`, such as age, sex, and biomarkers, including systolic blood pressure (BP). You can find more about these variables on the UK Biobank website ⬇️

: Help

Note: UK Biobank includes repeat measures for some variables, such as BP. The BP referred to here is the value at the time of imaging.

## Loading data

Pass an existing data folder to load its demographics. Point clouds are read
when requested and returned as their original NumPy arrays.
Training filenames contain the sample ID and MI label. The supplied test
filenames use CSV row positions (0–99), corresponding to IDs 900–999.

```python
from introml_myocardial_infarction import CardiacDataLoader

loader = CardiacDataLoader("data/training_data")
demographics = loader.demographics
print(demographics.head())

sample = loader[0]  # First row and its associated point cloud.
cloud = sample["point_cloud"]  # Shape: (10, 18000, 4).
xyz = cloud[..., :3]
surfaces = cloud[..., 3]

# Load a specific sample by its ID from the DataFrame index.
cloud = loader.load_point_cloud(demographics.index[0])

# All demographics plus a point_cloud column containing each full array.
data = loader.load_all()
cloud = data.loc[data.index[0], "point_cloud"]  # Shape: (10, 18000, 4).

# Select and flatten the coordinates in your analysis, for example for PCA.
import numpy as np

X = np.stack([cloud[0, :, :3].reshape(-1) for cloud in data["point_cloud"]])
# Shape: (900, 54000), with rows matching data.index.

test = CardiacDataLoader("data/test_samples")

# The labelled test demographics and test clouds are in separate folders.
labelled_test = CardiacDataLoader(
    "data/labelled_test_data",
    samples_dir="data/test_samples/test_samples",
)
```

`load_all()` memory maps the point clouds read-only by default, exposing every
frame without eagerly loading all arrays into RAM. Use
`loader.load_all(mmap_mode=None)` for fully loaded, writable arrays.

See `notebooks/pca.ipynb` for PCA of the first-frame coordinates, explained
variance, and 3D shape modes. With one frame per heart, these modes describe
variation between hearts. Anatomical interpretation assumes corresponding
points in a shared coordinate system across samples.

## Best geometric model: RBF score + demographics

The strongest model found so far uses heart geometry and patient demographics:

```bash
uv run python -m introml_myocardial_infarction.rbf_demographic_catboost \
  --output results/mi-rbf-demographics-catboost-stack-cv42
```

Each heart starts as 10 frames of 18,000 XYZ points, with a surface label for
each point. `geometry_features.py` converts that cloud into 702 numbers. For
each of the three surfaces (LV endocardium, LV epicardium, and RV) and each
frame, it subtracts that frame's whole-heart centre, then records the surface's
3D centroid, spread along each coordinate axis, three covariance eigenvalues,
and centroid change relative to frame 0. These are 12 shape and position values
per frame, or 120 per surface.

For movement, it measures each point's distance from its frame-0 position in
each frame, and the distance moved since the previous frame for each of the
nine transitions. For each set of distances it stores the 10th, 50th, and 90th
percentiles, mean, standard deviation, and maximum. That gives 10×6 + 9×6 =
114 motion values per surface. With 120 shape and position values, each surface
contributes 234 values; all three surfaces give 702 features. This summarizes
shape and movement without passing all 540,000 XYZ coordinates to the classifier.

An RBF SVM turns those summaries into a probability of MI. For training
patients, that probability is cross-fitted: each patient's score comes from a
model that did not train on that patient's label. CatBoost then combines this
score with seven numeric demographics (age, BMI, height, weight, blood
pressures and sex). Nested five-fold cross-validation selects settings and
thresholds using training data only. The resulting mean F1 was **0.909 ± 0.022**.
The full-training OOF score is used for selection and is optimistic; it is not
the independent CV estimate. Held-out test scores are excluded because the
mapping from test CSV rows to point-cloud files has not been verified.

The fitted training-only model is shared at
`models/rbf-demographic-catboost/model.joblib`. It contains the RBF SVM,
CatBoost model, selected classification threshold, and training sample IDs.
Its five-fold CV summary is in the same directory as `cv_summary.json`. The
checkpoint was fitted on the 897 retained training patients and does not use
the potentially mismatched test point clouds.
