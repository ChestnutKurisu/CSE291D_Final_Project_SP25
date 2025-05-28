import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

import cv2
import scipy.signal

from .simulator import SceneObject

XP = cp if cp is not None else np


class PointSource(SceneObject):
    def __init__(self, x, y, freq=0.1, amplitude=1.0):
        self.x = int(x)
        self.y = int(y)
        self.freq = freq
        self.amplitude = amplitude
        self.phase = 0.0

    def render(self, field, wave_speed_field, dampening_field):
        """Point sources leave the medium properties untouched."""
        pass  # field updates happen in :meth:`update_field`

    def update_field(self, field, t):
        if cp is not None and isinstance(field, cp.ndarray):
            field[self.y, self.x] += cp.sin(t * self.freq * 2 * cp.pi) * self.amplitude
        else:
            field[self.y, self.x] += np.sin(t * self.freq * 2 * np.pi) * self.amplitude

    def render_visualization(self, image):
        if 0 <= self.y < image.shape[0] and 0 <= self.x < image.shape[1]:
            cv2.circle(image, (self.x, self.y), 3, (50, 50, 50), -1)

class ConstantSpeed(SceneObject):
    def __init__(self, speed):
        self.speed = float(speed)

    def render(self, field, wave_speed_field, dampening_field):
        wave_speed_field[:] = self.speed

    def update_field(self, field, t):
        pass

    def render_visualization(self, image):
        pass


class StaticDampening(SceneObject):
    """Fixed dampening mask with optional absorbing border."""

    def __init__(self, dampening_field, border_thickness=0):
        arr = XP.clip(XP.asarray(dampening_field, dtype=XP.float32), 0.0, 1.0)
        self.d = arr.copy()
        h, w = self.d.shape
        for i in range(border_thickness):
            v = (i / border_thickness) ** 0.5
            self.d[i, i:w - i] = v
            self.d[h - 1 - i, i:w - i] = v
            self.d[i:h - i, i] = v
            self.d[i:h - i, w - 1 - i] = v

    def render(self, field, wave_speed_field, dampening_field):
        dampening_field[:] = self.d

    def update_field(self, field, t):
        pass

    def render_visualization(self, image):
        pass


class StaticRefractiveIndex(SceneObject):
    """Fixed refractive index field converted to wave speed."""

    def __init__(self, refractive_index_field):
        self.c = 1.0 / XP.clip(XP.asarray(refractive_index_field, dtype=XP.float32), 0.9, 10.0)

    def render(self, field, wave_speed_field, dampening_field):
        wave_speed_field[:] = self.c

    def update_field(self, field, t):
        pass

    def render_visualization(self, image):
        pass


class StaticImageScene(SceneObject):
    """Scene description via an image with RGB encoding."""

    def __init__(self, scene_image, source_amplitude=1.0, source_frequency_scale=1.0):
        self.source_opacity = 0.9
        self.refractive_index = StaticRefractiveIndex(scene_image[:, :, 0] / 100)
        self.dampening = StaticDampening(1.0 - scene_image[:, :, 2] / 255, border_thickness=48)

        sources_pos = np.flip(np.argwhere(scene_image[:, :, 1] > 0), axis=1)
        phase_amp_freq = np.tile(np.array([0, source_amplitude, 0.3]), (sources_pos.shape[0], 1))
        sources = np.concatenate((sources_pos, phase_amp_freq), axis=1)
        sources[:, 4] = scene_image[sources_pos[:, 1], sources_pos[:, 0], 1] / 255 * 0.5 * source_frequency_scale
        self.sources = XP.asarray(sources, dtype=XP.float32)

    def render(self, field, wave_speed_field, dampening_field):
        self.dampening.render(field, wave_speed_field, dampening_field)
        self.refractive_index.render(field, wave_speed_field, dampening_field)

    def update_field(self, field, t):
        xp = XP
        v = xp.sin(self.sources[:, 2] + self.sources[:, 4] * t) * self.sources[:, 3]
        coords = self.sources[:, 0:2].astype(xp.int32)
        o = self.source_opacity
        field[coords[:, 1], coords[:, 0]] = field[coords[:, 1], coords[:, 0]] * o + v * (1.0 - o)

    def render_visualization(self, image):
        pass


class StrainRefractiveIndex(SceneObject):
    """Refractive index field coupled to local strain."""

    def __init__(self, refractive_index_offset, coupling_constant):
        self.coupling_constant = coupling_constant
        self.refractive_index_offset = refractive_index_offset
        self.du_dx_kernel = XP.array([[-1, 0.0, 1]], dtype=XP.float32)
        self.du_dy_kernel = XP.array([[-1], [0.0], [1]], dtype=XP.float32)
        self.strain_field = None

    def render(self, field, wave_speed_field, dampening_field):
        xp = XP
        if cp is not None and xp is cp:
            import cupyx.scipy.signal as sig
            du_dx = sig.convolve2d(field, self.du_dx_kernel, mode="same", boundary="fill")
            du_dy = sig.convolve2d(field, self.du_dy_kernel, mode="same", boundary="fill")
        else:
            du_dx = scipy.signal.convolve2d(field, self.du_dx_kernel, mode="same", boundary="fill")
            du_dy = scipy.signal.convolve2d(field, self.du_dy_kernel, mode="same", boundary="fill")
        self.strain_field = xp.sqrt(du_dx ** 2 + du_dy ** 2)
        n_field = self.refractive_index_offset + self.strain_field * self.coupling_constant
        wave_speed_field[:] = 1.0 / xp.clip(n_field, 0.9, 10.0)

    def update_field(self, field, t):
        pass

    def render_visualization(self, image):
        pass


class StaticRefractiveIndexPolygon(SceneObject):
    """Anti-aliased polygon with constant refractive index."""

    def __init__(self, vertices, refractive_index):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.refractive_index = min(max(refractive_index, 0.9), 10.0)
        self._cached = None

    def _create_polygon_data(self, field_shape):
        if self._cached and self._cached[0] == field_shape:
            return self._cached[1]

        rows, cols = field_shape
        min_x = np.min(self.vertices[:, 0])
        max_x = np.max(self.vertices[:, 0])
        min_y = np.min(self.vertices[:, 1])
        max_y = np.max(self.vertices[:, 1])

        mask_w = int(np.ceil(max_x - min_x)) + 1
        mask_h = int(np.ceil(max_y - min_y)) + 1
        off_x = int(np.floor(min_x))
        off_y = int(np.floor(min_y))

        mask = np.zeros((mask_h, mask_w), dtype=np.float32)
        tv = np.round(self.vertices - [off_x, off_y]).astype(np.int32)
        cv2.fillPoly(mask, [tv], 1.0, lineType=cv2.LINE_AA)
        coords_y, coords_x = np.where(mask > 0)
        mask_values = mask[coords_y, coords_x]
        g_y = coords_y + off_y
        g_x = coords_x + off_x
        in_bounds = (g_y >= 0) & (g_y < rows) & (g_x >= 0) & (g_x < cols)
        valid_y = g_y[in_bounds]
        valid_x = g_x[in_bounds]
        valid_mask = mask_values[in_bounds]
        coords = (XP.asarray(valid_y), XP.asarray(valid_x))
        mask_values = XP.asarray(valid_mask, dtype=XP.float32)
        self._cached = (field_shape, (coords, mask_values))
        return coords, mask_values

    def render(self, field, wave_speed_field, dampening_field):
        coords, mask_values = self._create_polygon_data(wave_speed_field.shape)
        bg = wave_speed_field[coords[0], coords[1]]
        wave_speed_field[coords[0], coords[1]] = bg * (1.0 - mask_values) + mask_values / self.refractive_index

    def update_field(self, field, t):
        pass

    def render_visualization(self, image):
        vertices = np.round(self.vertices).astype(np.int32)
        cv2.fillPoly(image, [vertices], (60, 60, 60), lineType=cv2.LINE_AA)


class StaticRefractiveIndexBox(StaticRefractiveIndexPolygon):
    """Rotated rectangular refractive index region."""

    def __init__(self, center, box_size, box_angle_rad, refractive_index):
        cx, cy = center
        width, height = box_size
        half_w = width / 2.0
        half_h = height / 2.0
        local_vertices = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ], dtype=np.float32)
        rot = cv2.getRotationMatrix2D((0, 0), np.rad2deg(box_angle_rad), 1.0)
        rotated = cv2.transform(np.array([local_vertices]), rot)[0]
        translated = rotated + [cx, cy]
        super().__init__(translated, refractive_index)

