import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt


def _coadd_flux(wave, flux, ivar, mask=None, method="mean", weight=None):
    """ Coadd spectra using S/N weighted mean. """
    methods = ["mean", "median", "ivar-weighted-mean", "irms-weighted-mean"]
    if method not in methods:
        raise ValueError(f"Coadd type must be one of {methods}")
    if method == "mean":
        wl_weight = np.ones_like(flux) * np.expand_dims(weight, axis=-1)
        nan_mask = (np.isnan(flux)) | (np.isnan(wl_weight))
        flux[nan_mask] = 0.0
        wl_weight[nan_mask] = 1e-10 # hack
        stacked_flux = np.average(flux, weights=wl_weight, axis=0)
        stacked_flux[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        return stacked_flux
    elif method == "median":
        return np.nanmedian(flux, axis=0)
    elif method == "ivar-weighted-mean":
        wl_weight = ivar * np.expand_dims(weight, axis=-1)
        nan_mask = (np.isnan(flux)) | (np.isnan(wl_weight))
        flux[nan_mask] = 0.0
        wl_weight[nan_mask] = 1e-10 # hack
        stacked_flux = np.average(flux, weights=wl_weight, axis=0)
        stacked_flux[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        return stacked_flux
    elif method == "irms-weighted-mean":
        wl_weight = ivar**0.5 * np.expand_dims(weight, axis=-1)
        nan_mask = (np.isnan(flux)) | (np.isnan(wl_weight))
        flux[nan_mask] = 0.0
        wl_weight[nan_mask] = 1e-10 # hack
        stacked_flux = np.average(flux, weights=wl_weight, axis=0)
        stacked_flux[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        return stacked_flux

def bootstrap_coadd(wave, flux, ivar, mask=None, method="mean", weight=None):
    boot_idx = np.random.choice(
        np.arange(len(flux)), replace=True, size=np.max([10,int(len(flux)/10)])
    )  # take a random sample each iteration
    boot_coadd = _coadd_flux(
        wave, flux[boot_idx], ivar[boot_idx], mask=mask, method=method, weight=weight[boot_idx]
    )  # calculate the mean for each iteration
    #plt.plot(wave, boot_coadd)
    #plt.show()
    return boot_coadd


def coadd_flux(
    wave,
    flux,
    ivar,
    mask=None,
    method="mean",
    weight=None,
    stack_error="bootstrap",
    n_workers=1,
):
    if np.all(weight==None):
        weight = np.ones(len(flux))
    stack_errors = ["no-errors", "bootstrap"]
    if stack_error not in stack_errors:
        raise ValueError(f"Coadd type must be one of {stack_errors}")
    if stack_error == "no-errors":
        return _coadd_flux(wave, flux, ivar, mask=mask, method=method, weight=weight)
    if stack_error == "bootstrap":
        if n_workers == 1:
            boot_averages = np.array(
                [
                    bootstrap_coadd(
                        wave, flux, ivar, mask=mask, method=method, weight=weight
                    )
                    for _ in range(100)
                ]
            )
            return np.nanmean(boot_averages, axis=0), np.nanstd(boot_averages, axis=0)
        else:
            boot_averages = np.array(
                Parallel(n_jobs=n_workers)(
                    delayed(bootstrap_coadd)(
                        wave, flux, ivar, mask=mask, method=method, weight=weight
                    )
                    for _ in range(100)
                )
            )
            return np.nanmean(boot_averages, axis=0), np.nanstd(boot_averages, axis=0)
