import os
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np


from . import *


def stack_spectra(
    spectra=None,
    flux=None,
    wave=None,
    ivar=None,
    mask=None,
    redshift=None,
    fibermap=None,
    exp_fibermap=None,
    output_wave_grid=None,
    resample_resolution=0.08,
    resample_method="linear",
    norm_method="median",
    norm_flux_window=None,
    stack_redshfit=0.0,
    weight=None,
    stack_method="mean",
    stack_error="bootstrap",
    n_workers=-1,
):
    if spectra is None:
        if (flux is None) or (wave is None):
            raise ValueError(
                "Atleast flux and wave should be provided if spectra object not given"
            )
    if redshift is None:
        raise ValueError("Redshift array should be provided")

    # set the number of parallel workers
    if n_workers <= 0:
        n_workers = multiprocessing.cpu_count()
    else:
        n_workers = min(int(n_workers), multiprocessing.cpu_count())

    # unpack the desispec.spectra.Spectra object
    if spectra is not None:
        flux = spectra.flux
        wave = spectra.wave
        ivar = spectra.ivar
        mask = spectra.mask
        fibermap = spectra.fibermap
        exp_fibermap = spectra.exp_fibermap

    # Coadd cameras if needed
    if isinstance(flux, dict):
        flux, wave, ivar, mask = coadd_cameras(flux, wave, ivar, mask)
    # de-redshfit the spectra
    flux_dered = deredshift(flux, redshift, stack_redshfit, "flux")
    wave_dered = deredshift(wave, redshift, stack_redshfit, "wave")
    ivar_dered = deredshift(ivar, redshift, stack_redshfit, "ivar")

    # resample the spectra to a common grid
    if output_wave_grid is None:
        output_wave_grid = np.arange(
            np.min(wave_dered), np.max(wave_dered), resample_resolution
        )
    flux_grid, ivar_grid = resample(
        output_wave_grid,
        wave_dered,
        flux_dered,
        ivar_dered,
        fill_val=np.nan,
        method=resample_method,
        n_workers=n_workers,
    )
    # Normalize the spectra
    flux_normed, ivar_normed = normalize(
        output_wave_grid,
        flux_grid,
        ivar_grid,
        mask=mask,
        method=norm_method,
        flux_window=norm_flux_window,
        n_workers=n_workers,
    )

    # stack the spectra
    stacked_spectra = coadd_flux(
        output_wave_grid,
        flux_normed,
        ivar_normed,
        method=stack_method,
        n_workers=n_workers,
    )

    return stacked_spectra, output_wave_grid