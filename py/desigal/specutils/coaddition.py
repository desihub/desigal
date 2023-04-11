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
        nan_mask = (np.isnan(flux)) | (np.isnan(ivar)) | (ivar==0.) | (np.isnan(wl_weight))
        flux[nan_mask] = 0.0
        wl_weight[nan_mask] = 1e-10 # hack
        stacked_flux = np.average(flux, weights=wl_weight, axis=0)
        stacked_flux[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        stacked_ivar = np.average(ivar**-2, weights=wl_weight**2, axis=0)**-1
        stacked_ivar[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        return np.stack([stacked_flux, stacked_ivar], axis=-1)
    elif method == "median":
        return np.nanmedian(flux, axis=0)
    elif method == "ivar-weighted-mean":
        wl_weight = ivar * np.expand_dims(weight, axis=-1)
        nan_mask = (np.isnan(flux)) | (np.isnan(ivar)) | (ivar==0.) | (np.isnan(wl_weight))
        flux[nan_mask] = 0.0
        wl_weight[nan_mask] = 1e-10 # hack
        stacked_flux = np.average(flux, weights=wl_weight, axis=0)
        stacked_flux[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        stacked_ivar = np.average(ivar**-2, weights=wl_weight**2, axis=0)**-1
        stacked_ivar[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        return np.stack([stacked_flux, stacked_ivar], axis=-1)
    elif method == "irms-weighted-mean":
        wl_weight = ivar**0.5 * np.expand_dims(weight, axis=-1)
        nan_mask = (np.isnan(flux)) | (np.isnan(ivar)) | (ivar==0.) | (np.isnan(wl_weight))
        flux[nan_mask] = 0.0
        wl_weight[nan_mask] = 1e-10 # hack
        stacked_flux = np.average(flux, weights=wl_weight, axis=0)
        stacked_flux[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        stacked_ivar = np.average(ivar**-2, weights=wl_weight**2, axis=0)**-1
        stacked_ivar[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        return np.stack([stacked_flux, stacked_ivar], axis=-1)

def bootstrap_coadd(wave, flux, ivar, mask=None, method="mean", weight=None):
    boot_idx = np.random.choice(
        np.arange(len(flux)), replace=True, size=len(flux)
    )  # take a random sample each iteration
    boot_coadd = _coadd_flux(
        wave, 
        flux[boot_idx],#+np.random.normal(size=ivar[boot_idx].shape)*ivar[boot_idx]**-0.5, 
        ivar[boot_idx], 
        mask=mask, 
        method=method, 
        weight=weight[boot_idx]
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
    bootstrap_samples=1000,
):
    if np.all(weight==None):
        weight = np.ones(len(flux))
    stack_errors = ["propagated", "bootstrap"]
    if stack_error not in stack_errors:
        raise ValueError(f"Coadd type must be one of {stack_errors}")
    if stack_error == "propagated":
        return _coadd_flux(wave, flux, ivar, mask=mask, method=method, weight=weight)
    if stack_error == "bootstrap":
        print('start bootstrapping!')
        if n_workers == 1:
            boot_averages = np.array(
                [
                    bootstrap_coadd(
                        wave, flux, ivar, mask=mask, method=method, weight=weight
                    )
                    for _ in range(bootstrap_samples)
                ]
            )
            print(boot_averages.shape)
            return np.nanmean(boot_averages[:,:,0], axis=0), np.nanmean(boot_averages[:,:,1], axis=0)
        else:
            boot_averages = np.array(
                Parallel(n_jobs=n_workers)(
                    delayed(bootstrap_coadd)(
                        wave, flux, ivar, mask=mask, method=method, weight=weight
                    )
                    for _ in range(bootstrap_samples)
                )
            )
            print(boot_averages.shape)
            return np.nanmean(boot_averages[:,:,0], axis=0), np.nanmean(boot_averages[:,:,1], axis=0)
