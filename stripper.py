import skvideo.io
import numpy as np
from helper import *
from tqdm import tqdm

for movie in movies:
    print(movie)
    filename = "movies/%s.mp4" % movie
    metadata = skvideo.io.ffprobe(filename)
    framerate = metadata['video']['@avg_frame_rate']
    fr = framerate.split('/')
    fps = int(fr[0])/int(fr[1])
    spf = 1 / fps
    print('%.3f FPS' % fps)
    print('%.3f SPF' % spf)

    data = skvideo.io.vreader(filename)

    time = 0
    t_ctr = 0

    buffer = []
    colors = []
    for frame in tqdm(data):
        time += spf
        buffer.append(frame)

        # On full second
        if int(time) != t_ctr:
            t_ctr = int(time)

            buffer = np.array(buffer)
            color = np.mean(buffer, axis=(0,1,2)).astype(np.uint8)

            colors.append(color)

            buffer = []

    colors = np.array(colors)
    np.save('colors/%s' % movie, colors)
