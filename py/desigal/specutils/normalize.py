"""
possible normalization methods:
1. mean
2. median
3. flux window
4. continuum
5. iterative

NOTE: Masks currently marked by np.nan values
"""

from crypt import methods
import numpy as np
import numpy.ma as ma
from scipy import integrate, interpolate
from joblib import Parallel, delayed


def integrate_flux(wave, flux, ivar, w1, w2):
    """ Current assumption is fluxes are on same grid, need to make it more general """
    # trim for speed
    wpad = 2 * np.median(np.abs(np.diff(wave)))

    wave_mask = (wave > (w1 - wpad)) * (wave < (w2 + wpad))
    total_mask = np.all(
        [
            wave_mask,
            ivar > 0,
            np.isfinite(flux),
        ],
        axis=0,
    )

    # Require no more than 20% of pixels are masked.
    if np.sum(total_mask) / np.sum(wave_mask) < 0.8:
        print("More than 20% of pixels flux window are masked.")

    wave = wave[total_mask]
    flux = flux[total_mask]
    ivar = ivar[total_mask]
    # should never have to extrapolate
    f = interpolate.interp1d(wave, flux, bounds_error=True, axis=-1)
    f1 = f(w1)
    f2 = f(w2)
    i = interpolate.interp1d(wave, ivar, bounds_error=True, axis=-1)
    i1 = i(w1)
    i2 = i(w2)
    # insert the boundary wavelengths then integrate
    I = np.where((wave > w1) * (wave < w2))[0]
    wave = np.insert(wave[I], [0, len(I)], [w1, w2])
    flux = np.insert(flux[I], [0, len(I)], [f1, f2])
    ivar = np.insert(ivar[I], [0, len(I)], [i1, i2])
    weight = integrate.simps(x=wave, y=ivar)
    index = integrate.simps(x=wave, y=flux * ivar) / weight
    index_ivar = weight
    return index, index_ivar


def mean_normalize(wave, flux, ivar=None):

    norm = np.nanmean(flux, axis=-1)[:, None]
    if ivar is not None:
        return flux / norm, ivar * norm ** 2
    else:
        return flux / norm


def median_normalize(wave, flux, ivar=None):
    norm = np.nanmedian(flux, axis=-1)[:, None]
    if ivar is not None:
        return flux / norm, ivar * norm ** 2
    else:
        return flux / norm


def flux_window_normalize(wave, flux, ivar, flux_window, mask=None, n_workers=1):
    wpad = 2 * np.median(np.abs(np.diff(wave)))
    if np.min(wave) > (flux_window[0] - wpad) or np.max(wave) < (flux_window[1] + wpad):
        raise ValueError("Flux window is outside of wavelength range.")

    if n_workers == 1:
        norm, norm_ivar = map(
            np.array,
            zip(
                *[
                    integrate_flux(
                        wave, flux[i], ivar[i], flux_window[0], flux_window[1]
                    )
                    for i in range(len(flux))
                ]
            ),
        )
    else:
        norm, norm_ivar = map(
            np.array,
            zip(
                *Parallel(n_jobs=n_workers)(
                    delayed(integrate_flux)(
                        wave, flux[i], ivar[i], flux_window[0], flux_window[1]
                    )
                    for i in range(len(flux))
                )
            ),
        )

    return flux / norm[:, None], ivar * norm[:, None] ** 2


def continuum_normalize(wave, flux, ivar, mask=None):
    raise NotImplementedError


def iterative_normalize(wave, flux, ivar, mask=None):
    raise NotImplementedError


def normalize(wave, flux, ivar, mask=None, method="median", flux_window=None):
    if method not in ["mean", "median", "flux-window", "continuum", "iterative"]:
        raise ValueError(f"Unknown normalization method: {method}")
    if mask is not None:
        flux[mask] = np.nan
        ivar[mask] = np.nan
    if method == "mean":
        return mean_normalize(wave, flux, ivar)
    elif method == "median":
        return median_normalize(wave, flux, ivar)
    elif method == "flux-window":
        return flux_window_normalize(wave, flux, ivar, flux_window=flux_window)
    elif method == "continuum":
        return continuum_normalize(wave, flux, ivar)
    elif method == "iterative":
        return iterative_normalize(wave, flux, ivar)
