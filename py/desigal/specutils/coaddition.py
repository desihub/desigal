import numpy as np


def coadd_flux(wave, flux, ivar, mask=None, method="mean", weight=None, n_workers=1):
    """ Coadd spectra using S/N weighted mean. """
    methods = ["mean", "median", "ivar-weighted-mean", "irms-weighted-mean"]
    if np.all(weight==None):
        weight = np.ones(len(flux))
    if method not in methods:
        raise ValueError(f"Coadd type must be one of {methods}")
    if method == "mean":
        wl_weight = np.ones_like(flux) * np.expand_dims(weight, axis=-1)
        nan_mask = (np.isnan(flux)) | (np.isnan(wl_weight))
        flux[nan_mask] = 0.0
        wl_weight[nan_mask] = 0.0
        stacked_flux = np.average(flux, weights=wl_weight, axis=0)
        #stacked_flux[np.sum(nan_mask, axis=0)!=0] = np.nan # only allow wl which is covered by all spectra
        return stacked_flux
    elif method == "median":
        return np.nanmedian(flux, axis=0)
    elif method == "ivar-weighted-mean":
        wl_weight = ivar * np.expand_dims(weight, axis=-1)
        nan_mask = (np.isnan(flux)) | (np.isnan(wl_weight))
        flux[nan_mask] = 0.0
        wl_weight[nan_mask] = 0.0
        stacked_flux = np.average(flux, weights=wl_weight, axis=0)
        #stacked_flux[np.sum(nan_mask, axis=0)!=0] = np.nan # only allow wl which is covered by all spectra
        return stacked_flux
    elif method == "irms-weighted-mean":
        wl_weight = ivar**0.5 * np.expand_dims(weight, axis=-1)
        nan_mask = (np.isnan(flux)) | (np.isnan(wl_weight))
        flux[nan_mask] = 0.0
        wl_weight[nan_mask] = 0.0
        stacked_flux = np.average(flux, weights=wl_weight, axis=0)
        #stacked_flux[np.sum(nan_mask, axis=0)!=0] = np.nan # only allow wl which is covered by all spectra
        return stacked_flux
    


