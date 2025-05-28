import math
import imageio.v2 as imageio
import numpy as np


def collage_videos(paths, outfile, grid=None, fps=30):
    """Combine multiple video files into a tiled collage.

    Parameters
    ----------
    paths : list of str
        List of video file paths.
    outfile : str
        Output path for the collage video.
    grid : tuple of int, optional
        (rows, cols) layout.  If not given, a nearly square grid is chosen.
    fps : int, optional
        Frame rate of the output video.
    """
    readers = [imageio.get_reader(p) for p in paths]
    meta = readers[0].get_meta_data()
    if fps is None:
        fps = int(meta.get("fps", fps))
    nframes = min(r.count_frames() for r in readers)

    n = len(readers)
    if grid is None:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
    else:
        rows, cols = grid

    frame0 = readers[0].get_data(0)
    h, w = frame0.shape[:2]

    writer = imageio.get_writer(outfile, fps=fps)
    for i in range(nframes):
        tiles = []
        for r in readers:
            tiles.append(r.get_data(i))
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
