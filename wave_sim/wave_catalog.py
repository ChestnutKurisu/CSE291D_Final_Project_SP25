"""Placeholder catalog of many wave simulation classes."""
from .base import WaveSimulation

# List of (class_name, wave_speed) pairs. Speeds are arbitrary placeholders.
_wave_specs = [
    ("PrimaryWaveSimulation", 1.0),
    ("SecondaryWaveSimulation", 0.6),
    ("SHWaveSimulation", 0.7),
    ("SVWaveSimulation", 0.7),
    ("QuasiCompressionalWaveSimulation", 1.1),
    ("QuasiShear1WaveSimulation", 0.65),
    ("QuasiShear2WaveSimulation", 0.65),
    ("RayleighWaveSimulation", 0.8),
    ("LoveWaveSimulation", 0.75),
    ("StoneleyWaveSimulation", 0.7),
    ("ScholteWaveSimulation", 0.7),
    ("LeakyRayleighWaveSimulation", 0.8),
    ("HigherModeRayleighWaveSimulation", 0.8),
    ("HigherModeLoveWaveSimulation", 0.75),
    ("LambS0ModeSimulation", 0.9),
    ("LambA0ModeSimulation", 0.9),
    ("HigherOrderLambModeSimulation", 0.9),
    ("GuidedPSConvertedWaveSimulation", 0.85),
    ("BoreholeFlexuralWaveSimulation", 0.5),
    ("BoreholeScrewWaveSimulation", 0.5),
    ("BoreholeCompressionalTubeWaveSimulation", 0.55),
    ("LongitudinalRodWaveSimulation", 1.0),
    ("TorsionalShaftWaveSimulation", 0.8),
    ("FlexuralBeamWaveSimulation", 0.6),
    ("ThicknessShearPlateWaveSimulation", 0.6),
    ("SurfaceAcousticWaveSimulation", 0.9),
    ("BulkAcousticWaveSimulation", 1.1),
    ("AcoustoElasticWaveSimulation", 1.0),
    ("NonLinearSolitaryElasticWaveSimulation", 0.95),
    ("PhononicCrystalBlochWaveSimulation", 0.8),
    ("QuasiStaticCreepWaveSimulation", 0.4),
    ("PlaneAcousticWaveSimulation", 1.0),
    ("SphericalAcousticWaveSimulation", 1.0),
    ("ShockWaveSimulation", 1.2),
    ("NWaveSimulation", 1.1),
    ("HelmholtzResonatorStandingModeSimulation", 0.9),
    ("WhisperingGalleryAcousticModeSimulation", 0.9),
    ("ThermoAcousticWaveSimulation", 0.7),
    ("AcousticStreamingWaveSimulation", 0.7),
    ("InternalGravityWaveSimulation", 0.6),
    ("AcousticGravityAtmosphericWaveSimulation", 0.6),
    ("MachWaveSimulation", 1.3),
    ("DeepWaterGravityWaveSimulation", 0.8),
    ("ShallowWaterGravityWaveSimulation", 0.8),
    ("CapillaryWaveSimulation", 0.5),
    ("KelvinWaveSimulation", 0.5),
    ("RossbyWaveSimulation", 0.3),
    ("PoincareWaveSimulation", 0.6),
    ("TsunamiWaveSimulation", 0.9),
    ("SolitarySurfaceWaveSimulation", 0.8),
    ("TidalBoreSimulation", 0.7),
    ("SeicheWaveSimulation", 0.7),
    ("InternalSolitaryWaveSimulation", 0.6),
    ("DoubleDiffusiveConvectionWaveSimulation", 0.5),
    ("AlfvenWaveSimulation", 1.0),
    ("SlowMagnetoAcousticWaveSimulation", 0.9),
    ("FastMagnetoAcousticWaveSimulation", 1.1),
    ("MagnetoGravityWaveSimulation", 0.8),
    ("KelvinHelmholtzBillowWaveSimulation", 0.7),
    ("RadioWaveSimulation", 1.0),
    ("MicrowaveSimulation", 1.0),
    ("InfraRedWaveSimulation", 1.0),
    ("VisibleLightWaveSimulation", 1.0),
    ("UltraVioletWaveSimulation", 1.0),
    ("XRaySimulation", 1.0),
    ("GammaRaySimulation", 1.0),
    ("LangmuirWaveSimulation", 1.0),
    ("IonAcousticWaveSimulation", 0.8),
    ("HeatDiffusionWaveSimulation", 0.3),
    ("SpinWaveSimulation", 0.9),
]

__all__ = []

for class_name, speed in _wave_specs:
    def _factory(spd):
        class _Sim(WaveSimulation):
            def __init__(self, **kwargs):
                kwargs.setdefault('c', spd)
                super().__init__(**kwargs)
                self.initialize(amplitude=1.0)
        return _Sim
    cls = _factory(speed)
    cls.__name__ = class_name
    globals()[class_name] = cls
    __all__.append(class_name)
