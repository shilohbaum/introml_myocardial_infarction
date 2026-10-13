## Dataset ##

# About the data

The dataset contains cardiac anatomies represented as point clouds in the folder `mixed_samples`.

These point clouds are derived using the deep learning pipeline by Beetz et al. [1]. Each MR image is segmented and contours are stacked to form a sparse point cloud. A geometric deep learning model then aligns contours and produces a dense reconstruction of overall shape. Figure 1 outlines this procedure.

Figure 1- Point cloud reconstruction method presented by Beetz et al.

Typical cine MRI has 25 frames per cardiac cycle. UK Biobank has 50 frames. We provide 10 frames per sample to show the heart at 10 points in the cardiac cycle. Each 3D image has a resolution of $1.8\times1.8\times8$ mm.

The point clouds also have associated demographic characteristics in `mixed_demographics`, such as age, sex, and biomarkers, including systolic blood pressure (BP). You can find more about these variables on the UK Biobank website ⬇️

: Help

Note: UK Biobank includes repeat measures for some variables, such as BP. The BP referred to here is the value at the time of imaging.