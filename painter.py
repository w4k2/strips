import skvideo.io
import numpy as np
from helper import *
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.utils import resample

n_clusters = 16
depth = 4
slomo = 16

for movie in movies:
    print(movie)
    filename = "movies/%s.mp4" % movie
    metadata = skvideo.io.ffprobe(filename)
    framerate = metadata['video']['@avg_frame_rate']
    fr = framerate.split('/')
    fps = int(fr[0])/int(fr[1])
    spf = 1 / fps
    spf = spf * slomo
    print('%.3f FPS' % fps)
    print('%.3f SPF' % spf)

    data = skvideo.io.vreader(filename)

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
            X = X // depth
            X = X * depth

            # Divide to clusters
            clu = KMeans(n_clusters=n_clusters).fit(X)
            y_pred = clu.predict(X)

            # Gather them
            clusters, counts = np.unique(y_pred, return_counts=True)
            sorter = np.argsort(-counts)
            clusters = clusters[sorter]
            counts = counts[sorter]
            counts = counts / np.sum(counts)
            _counts = np.zeros(n_clusters)
            _counts[:counts.shape[0]] = counts[:counts.shape[0]]

            # Prepare palette
            palette = [np.mean(X[y_pred == cluster], axis=0).astype(np.uint8)
                       for cluster, count in zip(clusters, counts)]
            palette = np.array(palette)

            _palette = np.zeros((n_clusters, 3))
            _palette[:palette.shape[0]] = palette[:palette.shape[0]]

            # Generate frame output
            #print('palette', palette, palette.shape)
            #print('counts', counts, counts.shape)
            f_output = np.concatenate(
                (_palette, (255*_counts[:, np.newaxis]).astype(np.uint8)), axis=1
            ).astype(np.uint8)

            #print(f_output)

            painting.append(f_output)
            #print(f_output.shape)

            # Flush the past
            buffer = []

            #if t_ctr == 30:
            #    break

    painting = np.array(painting)

    print(painting.shape)
    np.save('transformed/%s' % movie, painting)
