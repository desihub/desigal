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
        ivar[nan_mask] = 1e-10 # hack
        wl_weight[nan_mask] = 1e-10 # hack
        stacked_flux = np.average(flux, weights=wl_weight, axis=0)
        stacked_flux[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        stacked_ivar = (np.sum(wl_weight**2 * ivar**-1, axis=0)/(np.sum(wl_weight, axis=0)**2))**-1
        stacked_ivar[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        return np.stack([stacked_flux, stacked_ivar], axis=-1)
    if method == "median":
        wl_weight = np.ones_like(flux) * np.expand_dims(weight, axis=-1)
        nan_mask = (np.isnan(flux)) | (np.isnan(ivar)) | (ivar==0.) | (np.isnan(wl_weight))
        flux[nan_mask] = 0.0
        ivar[nan_mask] = 1e-10 # hack
        wl_weight[nan_mask] = 1e-10 # hack
        stacked_flux = np.zeros((flux.shape[1]))
        for i in range(len(stacked_flux)):
            stacked_flux[i] = weighted_quantiles(flux[:,i], weights=wl_weight[:,i], quantiles=0.5)
        stacked_flux[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        stacked_ivar = (np.sum(wl_weight**2 * ivar**-1, axis=0)/(np.sum(wl_weight, axis=0)**2))**-1
        stacked_ivar[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        print(np.nansum(stacked_flux))
        return np.stack([stacked_flux, stacked_ivar], axis=-1)
    #elif method == "median":
    #    return np.nanmedian(flux, axis=0)
    elif method == "ivar-weighted-mean":
        wl_weight = ivar * np.expand_dims(weight, axis=-1)
        nan_mask = (np.isnan(flux)) | (np.isnan(ivar)) | (ivar==0.) | (np.isnan(wl_weight))
        flux[nan_mask] = 0.0
        ivar[nan_mask] = 1e-10 # hack
        wl_weight[nan_mask] = 1e-10 # hack
        stacked_flux = np.average(flux, weights=wl_weight, axis=0)
        stacked_flux[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        stacked_ivar = (np.sum(wl_weight**2 * ivar**-1, axis=0)/(np.sum(wl_weight, axis=0)**2))**-1
        stacked_ivar[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        return np.stack([stacked_flux, stacked_ivar], axis=-1)
    elif method == "irms-weighted-mean":
        wl_weight = ivar**0.5 * np.expand_dims(weight, axis=-1)
        nan_mask = (np.isnan(flux)) | (np.isnan(ivar)) | (ivar==0.) | (np.isnan(wl_weight))
        flux[nan_mask] = 0.0
        ivar[nan_mask] = 1e-10 # hack
        wl_weight[nan_mask] = 1e-10 # hack
        stacked_flux = np.average(flux, weights=wl_weight, axis=0)
        stacked_flux[np.sum(~nan_mask, axis=0)==0] = np.nan # only allow wl which is covered by at least 1 spectrum
        stacked_ivar = (np.sum(wl_weight**2 * ivar**-1, axis=0)/(np.sum(wl_weight, axis=0)**2))**-1
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
    bootstrap=True,
    bootstrap_samples=1000,
    n_workers=1
):
    if np.all(weight==None):
        weight = np.ones(len(flux))
    if bootstrap==False:
        stack = _coadd_flux(wave, flux, ivar, mask=mask, method=method, weight=weight)
        return stack[:,0], stack[:,1]
    if bootstrap==True:
        if n_workers == 1:
            boot_averages = np.array(
                [
                    bootstrap_coadd(
                        wave, flux, ivar, mask=mask, method=method, weight=weight
                    )
                    for _ in range(bootstrap_samples)
                ]
            )
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
            return np.nanmean(boot_averages[:,:,0], axis=0), np.nanmean(boot_averages[:,:,1], axis=0)

def weighted_quantiles(values, weights, quantiles=0.5, axis=0):
    i = np.argsort(values, axis=axis)
    c = np.cumsum(weights[i], axis=axis)
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]
