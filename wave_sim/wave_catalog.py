"""Expanded catalog of wave simulations.

This module defines a minimal :class:`~wave_sim.base.WaveSimulation` subclass
for each of seventy different wave types spanning seismic, acoustic, fluid,
electromagnetic and other phenomena.  The goal is illustrative rather than
physically complete: each class simply sets a default wave speed ``c`` and
initialises the field with a unit impulse in the centre of the grid.

In a real solver the governing equations and boundary conditions for many of
these waves would differ substantially.  Here we use the same 2‑D wave
equation for simplicity so that the classes mainly demonstrate how the
framework can be extended.
"""

from .base import WaveSimulation


##############################################################################
# SEISMIC BODY WAVES
##############################################################################


class PrimaryWave(WaveSimulation):
    """1. Primary (P) wave: compressional body wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class SecondaryWave(WaveSimulation):
    """2. Secondary (S) wave: shear body wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.6)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class SHWave(WaveSimulation):
    """3. SH wave: horizontally polarised shear."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.55)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class SVWave(WaveSimulation):
    """4. SV wave: vertically polarised shear."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.55)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class QuasiPWave(WaveSimulation):
    """5. Quasi-compressional (qP) wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.95)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class QuasiS1Wave(WaveSimulation):
    """6. Quasi-shear 1 (qS1) wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.58)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class QuasiS2Wave(WaveSimulation):
    """7. Quasi-shear 2 (qS2) wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.57)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


##############################################################################
# SEISMIC SURFACE & INTERFACE WAVES
##############################################################################


class RayleighWave(WaveSimulation):
    """8. Rayleigh surface wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.53)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class LoveWave(WaveSimulation):
    """9. Love surface wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.50)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class StoneleyWave(WaveSimulation):
    """10. Stoneley solid-solid interface wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.45)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class ScholteWave(WaveSimulation):
    """11. Scholte solid-fluid interface wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.40)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class LeakyRayleighWave(WaveSimulation):
    """12. Leaky Rayleigh wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.52)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class HigherModeRayleighWave(WaveSimulation):
    """13. Higher-mode Rayleigh wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.51)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class HigherModeLoveWave(WaveSimulation):
    """14. Higher-mode Love wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.48)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


##############################################################################
# SEISMIC & ELASTIC GUIDED WAVES
##############################################################################


class LambS0Mode(WaveSimulation):
    """15. Lamb S0 (symmetric) mode."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.65)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class LambA0Mode(WaveSimulation):
    """16. Lamb A0 (antisymmetric) mode."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.60)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class HigherOrderLambMode(WaveSimulation):
    """17. Higher-order Lamb modes."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.62)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class GuidedPSConvertedWave(WaveSimulation):
    """18. Guided P–S converted wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.70)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class BoreholeFlexuralWave(WaveSimulation):
    """19. Borehole flexural wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.30)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class BoreholeScrewWave(WaveSimulation):
    """20. Borehole screw (torsional) wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.35)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class BoreholeTubeWave(WaveSimulation):
    """21. Borehole tube wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.25)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


##############################################################################
# ELASTIC WAVES IN SOLIDS & STRUCTURES
##############################################################################


class LongitudinalRodWave(WaveSimulation):
    """22. Longitudinal rod wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.8)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class TorsionalShaftWave(WaveSimulation):
    """23. Torsional shaft wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.7)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class FlexuralBeamWave(WaveSimulation):
    """24. Flexural beam wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.4)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class ThicknessShearPlateWave(WaveSimulation):
    """25. Thickness-shear plate wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.55)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class SurfaceAcousticWave(WaveSimulation):
    """26. Surface acoustic wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.50)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class BulkAcousticWave(WaveSimulation):
    """27. Bulk acoustic wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.85)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class AcoustoElasticWave(WaveSimulation):
    """28. Acousto-elastic wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.75)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class NonlinearSolitaryElasticWave(WaveSimulation):
    """29. Non-linear solitary elastic wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.65)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class PhononicCrystalBlochWave(WaveSimulation):
    """30. Phononic crystal Bloch wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.60)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class QuasiStaticCreepWave(WaveSimulation):
    """31. Quasi-static creep wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.05)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


##############################################################################
# ACOUSTIC WAVES IN FLUIDS
##############################################################################


class PlaneAcousticWave(WaveSimulation):
    """32. Plane acoustic wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.9)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class SphericalAcousticWave(WaveSimulation):
    """33. Spherical acoustic wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.9)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class ShockWave(WaveSimulation):
    """34. Shock wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.1)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class NWave(WaveSimulation):
    """35. N-wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.2)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class HelmholtzResonatorWave(WaveSimulation):
    """36. Helmholtz resonator wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.3)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class WhisperingGalleryAcousticMode(WaveSimulation):
    """37. Whispering-gallery acoustic mode."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.25)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class ThermoAcousticWave(WaveSimulation):
    """38. Thermo-acoustic wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.8)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class AcousticStreamingWave(WaveSimulation):
    """39. Acoustic streaming wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.9)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class InternalGravityWave(WaveSimulation):
    """40. Internal gravity wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.2)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class AcousticGravityAtmosphericWave(WaveSimulation):
    """41. Acoustic-gravity atmospheric wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.35)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class MachSupersonicWave(WaveSimulation):
    """42. Mach (supersonic) wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.5)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


##############################################################################
# FLUID SURFACE & INTERNAL WAVES
##############################################################################


class DeepWaterGravityWave(WaveSimulation):
    """43. Deep-water gravity wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.5)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class ShallowWaterGravityWave(WaveSimulation):
    """44. Shallow-water gravity wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.4)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class CapillaryWave(WaveSimulation):
    """45. Capillary wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.3)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class KelvinWave(WaveSimulation):
    """46. Kelvin wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.2)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class RossbyPlanetaryWave(WaveSimulation):
    """47. Rossby (planetary) wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.05)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class PoincareWave(WaveSimulation):
    """48. Poincaré wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.25)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class TsunamiWave(WaveSimulation):
    """49. Tsunami wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class SolitarySolitonSurfaceWave(WaveSimulation):
    """50. Solitary (soliton) surface wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.45)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class TidalBoreWave(WaveSimulation):
    """51. Tidal bore."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.4)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class SeicheWave(WaveSimulation):
    """52. Seiche."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.35)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class InternalSolitaryWave(WaveSimulation):
    """53. Internal solitary wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.25)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class DoubleDiffusiveConvectionWave(WaveSimulation):
    """54. Double-diffusive convection wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.1)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


##############################################################################
# MAGNETO-HYDRODYNAMIC WAVES
##############################################################################


class AlfvenWave(WaveSimulation):
    """55. Alfvén wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class SlowMagnetoAcousticWave(WaveSimulation):
    """56. Slow magneto-acoustic wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.5)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class FastMagnetoAcousticWave(WaveSimulation):
    """57. Fast magneto-acoustic wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.2)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class MagnetoGravityWave(WaveSimulation):
    """58. Magneto-gravity wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.4)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class KelvinHelmholtzBillowWave(WaveSimulation):
    """59. Kelvin–Helmholtz billow wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.65)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


##############################################################################
# ELECTROMAGNETIC SPECTRUM WAVES
##############################################################################


class RadioWave(WaveSimulation):
    """60. Radio wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class Microwave(WaveSimulation):
    """61. Microwave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class InfraRedWave(WaveSimulation):
    """62. Infrared wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class VisibleLightWave(WaveSimulation):
    """63. Visible light wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class UltraVioletWave(WaveSimulation):
    """64. Ultraviolet wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class XRayWave(WaveSimulation):
    """65. X-ray wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class GammaRayWave(WaveSimulation):
    """66. Gamma ray wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


##############################################################################
# PLASMA & HIGH-FREQUENCY WAVES
##############################################################################


class LangmuirPlasmaWave(WaveSimulation):
    """67. Langmuir plasma wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.9)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class IonAcousticWave(WaveSimulation):
    """68. Ion-acoustic wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.2)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


##############################################################################
# OTHER CONTINUUM-WAVE PHENOMENA
##############################################################################


class HeatDiffusionWave(WaveSimulation):
    """69. Heat-diffusion wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.1)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class SpinMagnonWave(WaveSimulation):
    """70. Spin (magnon) wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.3)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


##############################################################################
# Export all class names for easy importing
##############################################################################


__all__ = [
    "PrimaryWave",
    "SecondaryWave",
    "SHWave",
    "SVWave",
    "QuasiPWave",
    "QuasiS1Wave",
    "QuasiS2Wave",
    "RayleighWave",
    "LoveWave",
    "StoneleyWave",
    "ScholteWave",
    "LeakyRayleighWave",
    "HigherModeRayleighWave",
    "HigherModeLoveWave",
    "LambS0Mode",
    "LambA0Mode",
    "HigherOrderLambMode",
    "GuidedPSConvertedWave",
    "BoreholeFlexuralWave",
    "BoreholeScrewWave",
    "BoreholeTubeWave",
    "LongitudinalRodWave",
    "TorsionalShaftWave",
    "FlexuralBeamWave",
    "ThicknessShearPlateWave",
    "SurfaceAcousticWave",
    "BulkAcousticWave",
    "AcoustoElasticWave",
    "NonlinearSolitaryElasticWave",
    "PhononicCrystalBlochWave",
    "QuasiStaticCreepWave",
    "PlaneAcousticWave",
    "SphericalAcousticWave",
    "ShockWave",
    "NWave",
    "HelmholtzResonatorWave",
    "WhisperingGalleryAcousticMode",
    "ThermoAcousticWave",
    "AcousticStreamingWave",
    "InternalGravityWave",
    "AcousticGravityAtmosphericWave",
    "MachSupersonicWave",
    "DeepWaterGravityWave",
    "ShallowWaterGravityWave",
    "CapillaryWave",
    "KelvinWave",
    "RossbyPlanetaryWave",
    "PoincareWave",
    "TsunamiWave",
    "SolitarySolitonSurfaceWave",
    "TidalBoreWave",
    "SeicheWave",
    "InternalSolitaryWave",
    "DoubleDiffusiveConvectionWave",
    "AlfvenWave",
    "SlowMagnetoAcousticWave",
    "FastMagnetoAcousticWave",
    "MagnetoGravityWave",
    "KelvinHelmholtzBillowWave",
    "RadioWave",
    "Microwave",
    "InfraRedWave",
    "VisibleLightWave",
    "UltraVioletWave",
    "XRayWave",
    "GammaRayWave",
    "LangmuirPlasmaWave",
    "IonAcousticWave",
    "HeatDiffusionWave",
    "SpinMagnonWave",
]

