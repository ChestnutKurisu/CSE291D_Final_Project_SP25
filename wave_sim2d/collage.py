import math
import imageio.v2 as imageio
import numpy as np
import cv2


def collage_videos(paths, outfile, grid=None, fps=30, out_width=1920, out_height=1080):
    readers = [imageio.get_reader(p) for p in paths]
    meta = readers[0].get_meta_data()
    if fps is None:
        fps = int(meta.get("fps", 30))
    nframes = min(r.count_frames() for r in readers)

    n = len(readers)
    if grid is None:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
    else:
        rows, cols = grid

    frame0 = readers[0].get_data(0)
    tile_w = out_width // cols
    tile_h = out_height // rows

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(outfile, fourcc, fps, (out_width, out_height))

    for i in range(nframes):
        tiles = []
        for r in readers:
            try:
                frame = r.get_data(i)
            except Exception:
                frame = np.zeros_like(frame0)
            frame = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            tiles.append(frame)
        while len(tiles) < rows * cols:
            tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
        row_imgs = []
        idx = 0
        for r_ in range(rows):
            row = np.concatenate(tiles[idx:idx+cols], axis=1)
            idx += cols
            row_imgs.append(row)
        collage_frame = np.concatenate(row_imgs, axis=0)
        writer.write(collage_frame.astype(np.uint8))

    writer.release()
    for r in readers:
        r.close()
