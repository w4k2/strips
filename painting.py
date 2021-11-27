
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from helper import *

fig, ax = plt.subplots(len(movies), 2, figsize=(10,10))

for m_idx, movie in enumerate(movies):
    print(movie)
    painting = np.load('transformed/%s.npy' % movie, allow_pickle=True).astype(np.uint8)
    print(painting.shape)

    print(painting)

    dynamics = np.array(painting[:,:,-1]).T
    rgb_painting = np.array(painting[:,:,:3]).swapaxes(0,1)

    cimg = resize(rgb_painting, (32, 128, 3))
    dimg = resize(dynamics.T, (32, 128))

    ax[m_idx,1].imshow(dynamics, interpolation='none',
                       aspect=(dynamics.shape[1]/3)/dynamics.shape[0],
                       cmap='binary_r')
    ax[m_idx,0].imshow(rgb_painting, vmin=0, vmax=255, interpolation='none',
                       aspect=(dynamics.shape[1]/3)/dynamics.shape[0])
    ax[m_idx,0].set_title('%s trailer' % movie)
    ax[m_idx,0].set_xticks([])
    ax[m_idx,0].set_yticks([])


plt.tight_layout()
plt.savefig('bar.png')
