import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from helper import *

fig, ax = plt.subplots(len(movies), 1, figsize=(10,10))

for m_idx, movie in enumerate(movies):
    print(movie)

    colors = np.load('colors/%s.npy' % movie)
    colors = np.array([colors])

    cimg = resize(colors, (32, 128, 3))

    ax[m_idx].imshow(cimg, vmin=0, vmax=255)
    ax[m_idx].set_title('%s trailer' % movie)
    ax[m_idx].set_xticks([])
    ax[m_idx].set_yticks([])

plt.savefig('foo.png')
