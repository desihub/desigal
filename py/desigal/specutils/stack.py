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
    stack_redshift=0.0,
    weight=None,
    stack_method="mean",
    bootstrap=True,
    bootstrap_samples=1000,
    n_workers=-1,
    return_normed_spectra=False,
    multiplication_factor = None,
    cosmo=None,
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
        flux, wave, ivar = coadd_cameras(flux, wave, ivar, mask)
    
    # Multiply spectra if wanted
    if multiplication_factor is not None:
        flux = flux*multiplication_factor
        ivar = ivar*multiplication_factor**-2
    
    # MW dust correct
    flux_mwcorr = mw_dust_correct(flux, wave, fibermap["TARGET_RA"], fibermap["TARGET_DEC"], "flux")
    ivar_mwcorr = mw_dust_correct(ivar, wave, fibermap["TARGET_RA"], fibermap["TARGET_DEC"], "ivar")

    # de-redshfit the spectra
    flux_dered = deredshift(flux_mwcorr, redshift, stack_redshift, "flux")
    wave_dered = deredshift(wave, redshift, stack_redshift, "wave")
    ivar_dered = deredshift(ivar_mwcorr, redshift, stack_redshift, "ivar")

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
    
    # Check if spectra can be normalized.
    wpad = 2 * np.median(np.abs(np.diff(output_wave_grid)))
    if np.min(output_wave_grid) > (norm_flux_window[0] - wpad) or np.max(output_wave_grid) < (norm_flux_window[1] + wpad):
        raise ValueError("Flux window is outside of wavelength range.")

    wave_mask = np.tile(np.expand_dims(
        (output_wave_grid > (norm_flux_window[0] - wpad)) * (output_wave_grid < (norm_flux_window[1] + wpad))
        , axis=0), (len(flux_grid),1))
    total_mask = np.all(
        [
            wave_mask,
            ivar_grid > 0,
            np.isfinite(flux_grid),
        ],
        axis=0,
    )
    if norm_method=='flux-window':
        norm_mask = np.sum(total_mask, axis=1) / np.sum(wave_mask, axis=1) > 0.8
        if np.any(~norm_mask):
            print('The following spectra were excluded as they could not be normalized: ', list(spectra.target_ids()[~norm_mask]))
    else:
        norm_mask = np.ones_like(np.sum(total_mask, axis=1), dtype='bool')
        
    # Normalize the spectra
    flux_normed, ivar_normed = normalize(
        output_wave_grid,
        flux_grid[norm_mask],
        ivar_grid[norm_mask],
        redshift=redshift[norm_mask],
        mask=mask,
        method=norm_method,
        flux_window=norm_flux_window,
        n_workers=n_workers,
        cosmo=cosmo,
    )

    # stack the spectra
    stacked_spectra = coadd_flux(
        output_wave_grid,
        flux_normed,
        ivar_normed,
        method=stack_method,
        weight=weight,
        bootstrap=bootstrap,
        bootstrap_samples=bootstrap_samples,
        n_workers=n_workers
    )
    if return_normed_spectra:
        return stacked_spectra, output_wave_grid, np.stack([flux_normed, ivar_normed])
    else:
        return stacked_spectra, output_wave_grid

def write_binned_stacks(
    outfile,
    wave,
    flux,
    ivar,
    resolution=None,
    stackids=None,
    stack_redshift=None,
    table_column_dict={},
    table_format_dict={},
):
    """
    Save spectra to a fits file compatible with FastSpecFit stackfit

    Parameters
    ----------
    outfile : str
        Path and file name to save the fits file.
    wave : np.array
        1-Dimensional numpy array of the wavelength values of the spectra
    flux : np.array
        2-Dimensional numpy array of the flux values
    ivar : np.array
        1-Dimensional numpy array of the ivar values
    resolution : np.array
        3-Dimensional numpy array of the resolution matrices of the spectra. Saved in the same format as in desispec.resolution.Resolution: array[nspec, ndiag, nwave]
    stackids : np.array
        1-Dimensional numpy array of unique stackids. Default np.arange(nspec)
    stack_redshift : np.array
        1-Dimensional numpy array of the redshifts of the stacks. Default np.zeros(nspec)
    table_column_dict : dict
        Dictionary with column names(keys) and data to be included in the fits file.
    table_format_dict : dict
        Dictionary with column names(keys) and formats of the columns to be included in the fits file.
    Returns
    -------
    Nothing
        Saves spectra to fits file.
    """
    from astropy.io import fits

    nobj, _ = flux.shape
    if np.all(stackids == None):
        stackids = np.arange(nobj)
    if np.all(stack_redshift == None):
        stack_redshift = np.zeros(nobj)

    hdulist = []

    hdr = fits.Header()
    hdr[
        "COMMENT"
    ] = "Stack file created using desihub.desigal function write_binned_stacks"
    empty_primary = fits.PrimaryHDU(header=hdr)
    hdulist.append(empty_primary)

    hduflux = fits.ImageHDU(flux.astype("f4"))
    hduflux.header["EXTNAME"] = "FLUX"
    hdulist.append(hduflux)

    hduivar = fits.ImageHDU(ivar.astype("f4"))
    hduivar.header["EXTNAME"] = "IVAR"
    hdulist.append(hduivar)

    hduwave = fits.ImageHDU(wave.astype("f8"))
    hduwave.header["EXTNAME"] = "WAVE"
    hduwave.header["BUNIT"] = "Angstrom"
    hduwave.header["AIRORVAC"] = ("vac", "vacuum wavelengths")
    hdulist.append(hduwave)

    if ~np.all(resolution == None):
        hdures = fits.ImageHDU(resolution.astype("f4"))
        hdures.header["EXTNAME"] = "RES"
        hdulist.append(hdures)

    c1 = fits.Column(name="STACKID", array=stackids, format="K")
    c2 = fits.Column(name="Z", array=stack_redshift, format="D")
    columns = [c1, c2]
    for key in table_column_dict.keys():
        if table_format_dict[key][0] == "P":
            columns.append(
                fits.Column(
                    name=key,
                    array=np.array(table_column_dict[key], dtype="object"),
                    format=table_format_dict[key],
                )
            )
        else:
            columns.append(
                fits.Column(
                    name=key,
                    array=table_column_dict[key],
                    format=table_format_dict[key],
                )
            )

    hdutable = fits.BinTableHDU.from_columns(columns)
    hdutable.header["EXTNAME"] = "STACKINFO"
    hdulist.append(hdutable)

    hx = fits.HDUList(hdulist)

    hx.writeto(outfile, overwrite=True, checksum=True)