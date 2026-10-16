from sklearn.cluster import KMeans
import os
import numpy as np

folder = "../Data/training_data/mixed_samples/"


# seed clsuters with verified clusters
sample0_filepath = "../Data/training_data/mixed_samples/0healthy.npy"
sample0 = np.load(sample0_filepath)

sample113_filepath = "../Data/training_data/mixed_samples/113pMI.npy"
sample113 = np.load(sample113_filepath)

FRAMES = 10

train = os.listdir(folder)

frames_concat = np.ndarray((len(train) * (FRAMES), 72000))

for i, file in enumerate(train):
    sample = np.load(os.path.join(folder, file))

    for j in range(FRAMES):
        frame0 = sample[j, :, :]
        frames_concat[i * (FRAMES) + j] = frame0.flatten()


kmeans = KMeans(n_clusters=2, tol=1e-8, init=np.array([
    sample0[0, :, :].flatten(),
    sample113[0, :, :].flatten()
])).fit(frames_concat)


print("Error frames are at indices: ")
print(np.argwhere(kmeans.labels_))


