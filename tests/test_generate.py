import tempfile
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from examples.alfven_wave import generate_animation as gen_alfven
from examples.flexural_beam_wave import generate_animation as gen_flexural
from examples.internal_gravity_wave import generate_animation as gen_internal
from examples.kelvin_wave import generate_animation as gen_kelvin
from examples.rossby_planetary_wave import generate_animation as gen_rossby
from examples.plane_acoustic_wave import generate_animation as gen_plane
from examples.spherical_acoustic_wave import generate_animation as gen_spherical
from examples.deep_water_gravity_wave import generate_animation as gen_deep
from examples.shallow_water_gravity_wave import generate_animation as gen_shallow
from examples.capillary_wave import generate_animation as gen_capillary

GENERATORS = [
    gen_alfven,
    gen_flexural,
    gen_internal,
    gen_kelvin,
    gen_rossby,
    gen_plane,
    gen_spherical,
    gen_deep,
    gen_shallow,
    gen_capillary,
]

def test_generate_dry_run():
    out_dir = tempfile.mkdtemp()
    for gen in GENERATORS:
        try:
            path = gen(output_dir=out_dir, out_name="tmp.mp4", steps=2)
        except Exception:
            path = None
        assert isinstance(path, str)
        assert os.path.exists(out_dir)
