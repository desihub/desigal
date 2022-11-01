import numpy as np


def coadd_flux(wave, flux, ivar, mask=None, method="mean", n_workers=1):
    """ Coadd spectra using S/N weighted mean. """
    methods = ["mean", "median", "ivar-weighted-mean"]
    if method not in methods:
        raise ValueError(f"Coadd type must be one of {methods}")
    if method == "mean":
        return np.nanmean(flux, axis=0)
    elif method == "median":
        return np.nanmedian(flux, axis=0)
    elif method == "ivar-weighted-mean":
        raise NotImplementedError
