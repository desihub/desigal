import numpy as np


def coadd_flux(wave, flux, ivar, mask=None, method="mean", weight=None, n_workers=1):
    """ Coadd spectra using S/N weighted mean. """
    methods = ["mean", "median", "ivar-weighted-mean", "spec-weighted-mean"]
    if method not in methods:
        raise ValueError(f"Coadd type must be one of {methods}")
    if method == "mean":
        return np.nanmean(flux, axis=0)
    elif method == "median":
        return np.nanmedian(flux, axis=0)
    elif method == "ivar-weighted-mean":
        weight = ivar
        nan_mask = (np.isnan(flux)) | (np.isnan(weight))
        flux[nan_mask] = 0.0
        weight[nan_mask] = 0.0
        return np.average(flux, weights=weight, axis=0)
    elif method == "spec-weighted-mean":
        if np.any(weight==None):
            raise ValueError(f"Weight must be specified when performing a weighted average.")
        nan_mask = (np.isnan(flux))
        #flux[nan_mask] = 0.0
        return np.average(flux, weights=weight, axis=0)
