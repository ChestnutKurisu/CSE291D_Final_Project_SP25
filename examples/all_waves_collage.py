"""Generate short animations for all wave classes.

This script iterates over every wave simulation defined in
``wave_sim.wave_catalog`` and saves a small MP4 animation for each one in an
``output`` directory.  The purpose is to demonstrate usage of the expanded
wave catalog rather than to provide high fidelity results.
"""

import os
import numpy as np

import matplotlib

matplotlib.use("Agg")  # allow running without a display

from wave_sim import (
    PrimaryWave,
    SecondaryWave,
    SHWave,
    SVWave,
    QuasiPWave,
    QuasiS1Wave,
    QuasiS2Wave,
    RayleighWave,
    LoveWave,
    StoneleyWave,
    ScholteWave,
    LeakyRayleighWave,
    HigherModeRayleighWave,
    HigherModeLoveWave,
    LambS0Mode,
    LambA0Mode,
    HigherOrderLambMode,
    GuidedPSConvertedWave,
    BoreholeFlexuralWave,
    BoreholeScrewWave,
    BoreholeTubeWave,
    LongitudinalRodWave,
    TorsionalShaftWave,
    FlexuralBeamWave,
    ThicknessShearPlateWave,
    SurfaceAcousticWave,
    BulkAcousticWave,
    AcoustoElasticWave,
    NonlinearSolitaryElasticWave,
    PhononicCrystalBlochWave,
    QuasiStaticCreepWave,
    PlaneAcousticWave,
    SphericalAcousticWave,
    ShockWave,
    NWave,
    HelmholtzResonatorWave,
    WhisperingGalleryAcousticMode,
    ThermoAcousticWave,
    AcousticStreamingWave,
    InternalGravityWave,
    AcousticGravityAtmosphericWave,
    MachSupersonicWave,
    DeepWaterGravityWave,
    ShallowWaterGravityWave,
    CapillaryWave,
    KelvinWave,
    RossbyPlanetaryWave,
    PoincareWave,
    TsunamiWave,
    SolitarySolitonSurfaceWave,
    TidalBoreWave,
    SeicheWave,
    InternalSolitaryWave,
    DoubleDiffusiveConvectionWave,
    AlfvenWave,
    SlowMagnetoAcousticWave,
    FastMagnetoAcousticWave,
    MagnetoGravityWave,
    KelvinHelmholtzBillowWave,
    RadioWave,
    Microwave,
    InfraRedWave,
    VisibleLightWave,
    UltraVioletWave,
    XRayWave,
    GammaRayWave,
    LangmuirPlasmaWave,
    IonAcousticWave,
    HeatDiffusionWave,
    SpinMagnonWave,
)


def gaussian_source(X, Y, sigma=5.0):
    cx = X.shape[0] // 2
    cy = Y.shape[1] // 2
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


def run_and_save(sim, name, steps=50):
    ani = sim.animate(steps=steps)
    path = f"{name}.mp4"
    ani.save(path)
    return path


def main():
    os.makedirs("output", exist_ok=True)

    wave_classes = [
        PrimaryWave,
        SecondaryWave,
        SHWave,
        SVWave,
        QuasiPWave,
        QuasiS1Wave,
        QuasiS2Wave,
        RayleighWave,
        LoveWave,
        StoneleyWave,
        ScholteWave,
        LeakyRayleighWave,
        HigherModeRayleighWave,
        HigherModeLoveWave,
        LambS0Mode,
        LambA0Mode,
        HigherOrderLambMode,
        GuidedPSConvertedWave,
        BoreholeFlexuralWave,
        BoreholeScrewWave,
        BoreholeTubeWave,
        LongitudinalRodWave,
        TorsionalShaftWave,
        FlexuralBeamWave,
        ThicknessShearPlateWave,
        SurfaceAcousticWave,
        BulkAcousticWave,
        AcoustoElasticWave,
        NonlinearSolitaryElasticWave,
        PhononicCrystalBlochWave,
        QuasiStaticCreepWave,
        PlaneAcousticWave,
        SphericalAcousticWave,
        ShockWave,
        NWave,
        HelmholtzResonatorWave,
        WhisperingGalleryAcousticMode,
        ThermoAcousticWave,
        AcousticStreamingWave,
        InternalGravityWave,
        AcousticGravityAtmosphericWave,
        MachSupersonicWave,
        DeepWaterGravityWave,
        ShallowWaterGravityWave,
        CapillaryWave,
        KelvinWave,
        RossbyPlanetaryWave,
        PoincareWave,
        TsunamiWave,
        SolitarySolitonSurfaceWave,
        TidalBoreWave,
        SeicheWave,
        InternalSolitaryWave,
        DoubleDiffusiveConvectionWave,
        AlfvenWave,
        SlowMagnetoAcousticWave,
        FastMagnetoAcousticWave,
        MagnetoGravityWave,
        KelvinHelmholtzBillowWave,
        RadioWave,
        Microwave,
        InfraRedWave,
        VisibleLightWave,
        UltraVioletWave,
        XRayWave,
        GammaRayWave,
        LangmuirPlasmaWave,
        IonAcousticWave,
        HeatDiffusionWave,
        SpinMagnonWave,
    ]

    files = []
    for wave_cls in wave_classes:
        name = wave_cls.__name__
        print(f"Running {name}...")
        sim = wave_cls(grid_size=100, boundary="absorbing", source_func=gaussian_source)
        outfile = os.path.join("output", name.lower())
        files.append(run_and_save(sim, outfile, steps=50))

    print("Generated files:")
    for path in files:
        print("  ", path)


if __name__ == "__main__":
    main()

