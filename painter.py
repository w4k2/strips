import skvideo.io
import numpy as np
from helper import *
from method import Paletogram

for movie in movies:
    print(movie)
    filename = "movies/%s.mp4" % movie
    paletogram = Paletogram(filename)
    paletogram.process()
    np.save('transformed/%s' % movie, paletogram.painting)
