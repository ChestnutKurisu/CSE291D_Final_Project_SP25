import math
import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import zoom


def collage_videos(paths, outfile, grid=None, fps=30, mode="grid", scale=1.0):
    """Combine multiple video files into a tiled collage.

    Parameters
    ----------
    paths : list of str
        List of video file paths.
    outfile : str
        Output path for the collage video.
    grid : tuple of int, optional
        (rows, cols) layout for ``mode='grid'``. If not given, a nearly square
        grid is chosen.
    fps : int, optional
        Frame rate of the output video.
    mode : {"grid", "horizontal", "vertical", "overlay"}, optional
        Layout mode. ``horizontal`` and ``vertical`` ignore ``grid``. ``overlay``
        averages all frames into a single image.
    scale : float, optional
        Scale factor applied to each frame before compositing.
    """
    readers = [imageio.get_reader(p) for p in paths]
    meta = readers[0].get_meta_data()
    if fps is None:
        fps = int(meta.get("fps", fps))
    nframes = min(r.count_frames() for r in readers)

    n = len(readers)
    if mode == "horizontal":
        rows, cols = 1, n
    elif mode == "vertical":
        rows, cols = n, 1
    else:
        if grid is None:
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
        else:
            rows, cols = grid

    frame0 = readers[0].get_data(0)
    if scale != 1.0:
        frame0 = zoom(frame0, (scale, scale, 1), order=1)
    h, w = frame0.shape[:2]
    if mode == "overlay":
        rows = cols = 1

    writer = imageio.get_writer(outfile, fps=fps)
    for i in range(nframes):
        tiles = []
        for r in readers:
            frame = r.get_data(i)
            if scale != 1.0:
                frame = zoom(frame, (scale, scale, 1), order=1)
            tiles.append(frame)

        if mode == "overlay":
            accum = np.zeros_like(frame0, dtype=np.float32)
            for frame in tiles:
                accum += frame.astype(np.float32) / len(tiles)
            collage = accum.astype(np.uint8)
        else:
            # pad list if grid larger than number of videos
            while len(tiles) < rows * cols:
                tiles.append(np.zeros_like(frame0))

            row_imgs = []
            for r in range(rows):
                row = np.concatenate(tiles[r*cols:(r+1)*cols], axis=1)
                row_imgs.append(row)
            collage = np.concatenate(row_imgs, axis=0)
        writer.append_data(collage)
    writer.close()
    for r in readers:
        r.close()
