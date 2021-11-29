import skvideo.io
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.cluster import KMeans

class Paletogram:
    def __init__(self, filepath, n_clusters=16, depth=4, slomo=1):
        self.filepath = filepath
        self.n_clusters = n_clusters
        self.depth = depth
        self.slomo = slomo

    def process(self):
        metadata = skvideo.io.ffprobe(self.filepath)
        framerate = metadata['video']['@avg_frame_rate']
        fr = framerate.split('/')
        fps = int(fr[0])/int(fr[1])
        spf = 1 / fps
        spf = spf * self.slomo

        data = skvideo.io.vreader(self.filepath)

        time = 0
        t_ctr = 0

        buffer = []
        colors = []
        cube = []
        painting = []

        for frame in tqdm(data):
            time += spf
            buffer.append(frame)

            # On full second
            if int(time) != t_ctr:
                t_ctr = int(time)

            # Reconstruct
            buffer = np.array(buffer)
            flatten = buffer.reshape(-1, 3)
            X = resample(flatten, n_samples=1000, random_state=1410)
            X = X * self.depth

            # Divide to clusters
            clu = KMeans(n_clusters=self.n_clusters).fit(X)
            y_pred = clu.predict(X)

            # Gather them
            clusters, counts = np.unique(y_pred, return_counts=True)
            sorter = np.argsort(-counts)
            clusters = clusters[sorter]
            counts = counts[sorter]
            counts = counts / np.sum(counts)
            _counts = np.zeros(self.n_clusters)
            _counts[:counts.shape[0]] = counts[:counts.shape[0]]

            # Prepare palette
            palette = [np.mean(X[y_pred == cluster], axis=0).astype(np.uint8)
                       for cluster, count in zip(clusters, counts)]
            palette = np.array(palette)

            _palette = np.zeros((self.n_clusters, 3))
            _palette[:palette.shape[0]] = palette[:palette.shape[0]]

            # Generate frame output
            f_output = np.concatenate(
                (_palette, (255*_counts[:, np.newaxis]).astype(np.uint8)), axis=1
            ).astype(np.uint8)

            painting.append(f_output)

            # Flush the past
            buffer = []

        self.painting = np.array(painting)
