
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from helper import *
from scipy.signal import convolve2d

fig, ax = plt.subplots(len(movies), figsize=(10,8))

for m_idx, movie in enumerate(movies):
    print(movie)
    painting = np.load('transformed/%s.npy' % movie, allow_pickle=True).astype(np.uint8)
    print(painting.shape)

    print(painting)

    dynamics = np.array(painting[:,:,-1]).T
    rgb_painting = np.array(painting[:,:,:3]).swapaxes(0,1)

    kernel = np.array([
        [0,1,2,1,0],
        [0,1,2,1,0]
    ])
    kernel = kernel / np.sum(kernel)

    #dynamics = convolve2d(dynamics, kernel, mode='same')
    #for c in range(3):
    #    rgb_painting[:,:,c] = convolve2d(rgb_painting[:,:,c], kernel, mode='same', boundary='symm')


    cimg = resize(rgb_painting, (32, 128, 3))
    dimg = resize(dynamics.T, (32, 128))

    #ax[m_idx,1].imshow(dynamics, interpolation='none',
    #                   aspect=(dynamics.shape[1]/10)/dynamics.shape[0],
    #                   cmap='binary_r')
    ax[m_idx].imshow(rgb_painting, vmin=0, vmax=255, interpolation='none',
                       aspect=(dynamics.shape[1]/5)/dynamics.shape[0])
    ax[m_idx].set_title('%s' % movie)
    ax[m_idx].set_xticks([])
    ax[m_idx].set_yticks([])
    #ax[m_idx,1].set_xticks([])
    #ax[m_idx,1].set_yticks([])


plt.tight_layout()
plt.savefig('bar.png')
