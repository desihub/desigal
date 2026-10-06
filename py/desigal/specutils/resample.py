""" Resample a spectrum to a new wavelength grid.
Currently supported methods are:
1. Linear interpolated resampler
2. S/N conserving resampler
3. Flux conserving resampler
4. Spectroperfection resampler

1 is fastest, 2 and 3 are relatively fast courtesy of parallelization and 4 is slow.
1,2,3 will provide correlated wavelength bins while 4 is not flux conserving.
"""
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import desispec
from desispec.interpolation import resample_flux


def linear_resample(wave_new, wave, flux, ivar=None, fill_val=np.nan, n_workers=1):
    """
    Resample a spectrum to a new wavelength grid using linear interpolation.
    """

    if n_workers == 1:
        flux_new = np.array(
            [
                np.interp(wave_new, wave[i], flux[i], left=fill_val, right=fill_val)
                for i in range(len(flux))
            ]
        )
        if ivar is not None:
            ivar_new = np.array(
                [
                    np.interp(wave_new, wave[i], ivar[i], left=fill_val, right=fill_val)
                    for i in range(len(flux))
                ]
            )
            return flux_new, ivar_new
        return flux_new
    else:
        flux_new = np.array(
            Parallel(n_jobs=n_workers)(
                delayed(np.interp)(
                    wave_new, wave[i], flux[i], left=fill_val, right=fill_val
                )
                for i in range(len(flux))
            )
        )
        if ivar is not None:
            ivar_new = np.array(
                Parallel(n_jobs=n_workers)(
                    delayed(np.interp)(
                        wave_new, wave[i], ivar[i], left=fill_val, right=fill_val
                    )
                    for i in range(len(flux))
                )
            )
            return np.array(flux_new), np.array(ivar_new)
        return flux_new

# Vendored verbatim from desispec/quicklook/palib.py as of
# desihub/desispec@38acb7f, the last commit before desispec.quicklook was
# retired in desihub/desispec@c8d7721 ("move unused code", 2024-10-17).
# Only the name has changed (project -> _project). Both projects are BSD-3.
def _project(x1, x2):
    """
    return a projection matrix so that arrays are related by linear interpolation
    x1: Array with one binning
    x2: new binning

    Return Pr: x1= Pr.dot(x2) in the overlap region
    """
    x1=np.sort(x1)
    x2=np.sort(x2)
    Pr=np.zeros((len(x2),len(x1)))

    e1 = np.zeros(len(x1)+1)
    e1[1:-1]=(x1[:-1]+x1[1:])/2.0  # calculate bin edges
    e1[0]=1.5*x1[0]-0.5*x1[1]
    e1[-1]=1.5*x1[-1]-0.5*x1[-2]
    e1lo = e1[:-1]  # make upper and lower bounds arrays vs. index
    e1hi = e1[1:]

    e2=np.zeros(len(x2)+1)
    e2[1:-1]=(x2[:-1]+x2[1:])/2.0  # bin edges for resampled grid
    e2[0]=1.5*x2[0]-0.5*x2[1]
    e2[-1]=1.5*x2[-1]-0.5*x2[-2]

    for ii in range(len(e2)-1): # columns
        #- Find indices in x1, containing the element in x2
        #- This is much faster than looping over rows

        k = np.where((e1lo<=e2[ii]) & (e1hi>e2[ii]))[0]
        # this where obtains single e1 edge just below start of e2 bin
        emin = e2[ii]
        emax = e1hi[k]
        if e2[ii+1] < emax : emax = e2[ii+1]
        dx = (emax-emin)/(e1hi[k]-e1lo[k])
        Pr[ii,k] = dx    # enter first e1 contribution to e2[ii]

        if e2[ii+1] > emax :
            # cross over to another e1 bin contributing to this e2 bin
            l = np.where((e1 < e2[ii+1]) & (e1 > e1hi[k]))[0]
            if len(l) > 0 :
               # several-to-one resample.  Just consider 3 bins max. case
               Pr[ii,k[0]+1] = 1.0  # middle bin fully contained in e2
               q = k[0]+2
            else : q = k[0]+1  # point to bin partially contained in current e2 bin

            try:
                emin = e1lo[q]
                emax = e2[ii+1]
                dx = (emax-emin)/(e1hi[q]-e1lo[q])
                Pr[ii,q] = dx
            except:
                pass

    #- edge:
    if x2[-1]==x1[-1]:
        Pr[-1,-1]=1
    return Pr


def _bin_edges(x):
    """Bin edges for a sample grid, using the same convention as _project."""
    e = np.zeros(len(x) + 1)
    e[1:-1] = (x[:-1] + x[1:]) / 2.0
    e[0] = 1.5 * x[0] - 0.5 * x[1]
    e[-1] = 1.5 * x[-1] - 0.5 * x[-2]
    return e


def _sn_conserving_resample(wave,flux,outwave,ivar=None):

    # Not part of the vendored body: _project assumes every output bin edge
    # falls inside the input grid, and silently indexes an empty array
    # ("ValueError: The truth value of an empty array is ambiguous") when it
    # does not. Fail with something actionable instead.
    e_in, e_out = _bin_edges(np.sort(wave)), _bin_edges(np.sort(outwave))
    if e_out[0] < e_in[0] or e_out[-1] > e_in[-1]:
        raise ValueError(
            "sn-cons resampling requires the output grid to lie within the "
            "input grid: output bin edges span "
            f"[{e_out[0]:.4f}, {e_out[-1]:.4f}] but the input grid only covers "
            f"[{e_in[0]:.4f}, {e_in[-1]:.4f}]. Note the edges extend half a bin "
            "beyond the first and last samples. Use method='flux-cons' or "
            "'linear' if you need the output grid to overhang the input."
        )

    #- convert flux to per bin before projecting to new bins
    flux=flux*np.gradient(wave)

    Pr=_project(wave,outwave)
    n=len(wave)
    newflux=Pr.dot(flux)
    #- convert back to df/dx (per angstrom) sampled at outwave
    newflux/=np.gradient(outwave) #- per angstrom
    if ivar is None:
        return newflux
    else:
        ivar = ivar/(np.gradient(wave))**2.0

        # Not part of the vendored body: pixels with no data (ivar <= 0) have
        # infinite variance, and feeding 1/ivar = inf into the projection makes
        # 0 * inf = nan for every zero entry of Pr. Pr is mostly zeros, so a
        # single masked input pixel turned almost the whole output ivar into
        # nan -- and real DESI coadds always carry masked pixels.
        #
        # Keep the variance array finite, and instead flag the output bins that
        # draw any weight from a no-data pixel. Those bins genuinely contain a
        # gap, so they get ivar = 0 ("no information") rather than a number
        # computed from data that is not there. Using ivar[good] ** (-1.0)
        # rather than 1.0 / ivar[good] keeps the all-good case bit-identical to
        # the original expression.
        nodata = ~(ivar > 0)
        var = np.zeros_like(ivar)
        var[~nodata] = ivar[~nodata] ** (-1.0)
        newvar = Pr.dot(var) #- maintaining Total S/N
        contaminated = Pr[:, nodata].sum(axis=1) > 0
        # RK:  this is just a kludge until we more robustly ensure newvar is correct
        k = np.where(newvar <= 0.0)[0]
        newvar[k] = 0.0000001  # flag bins with no contribution from input grid
        newivar=1/newvar
        # newivar[k] = 0.0

        #- convert to per angstrom
        newivar*=(np.gradient(outwave))**2.0
        # Output bins overlapping a no-data input pixel carry no information.
        newivar[contaminated] = 0.0
        return newflux, newivar

def sn_conserving_resample(
    wave_new, wave, flux, ivar=None, fill_val=np.nan, verbose=False, n_workers=1
):
    """
    Resample a spectrum to a new wavelength grid using S/N conserving resampling.
    
    
    Algorithm is based on http://www.ast.cam.ac.uk/%7Erfc/vpfit10.2.pdf
    Appendix: B.1

    Args:
    wave : original wavelength array (expected (but not limited) to be native CCD pixel wavelength grid
    wave_new: new wavelength array: expected (but not limited) to be uniform binning
    flux : df/dx (Flux per A) sampled at x
    ivar : ivar in original binning. If not None, ivar in new binning is returned.

    Note:
    Full resolution computation for resampling is expensive for quick look.

    desispec.interpolation.resample_flux using weights by ivar does not conserve total S/N.
    Tests with arc lines show much narrow spectral profile, thus not giving realistic psf resolutions
    This algorithm gives the same resolution as obtained for native CCD binning, i.e, resampling has
    insignificant effect. Details,plots in the arc processing note.
    
    """
    if n_workers == 1:
        flux_new, ivar_new = zip(
            *[
                _sn_conserving_resample(
                    wave=wave[i], flux=flux[i], outwave=wave_new, ivar=ivar[i]
                )
                for i in range(len(flux))
            ]
        )
    else:
        flux_new, ivar_new = zip(
            *Parallel(n_jobs=n_workers)(
                delayed(_sn_conserving_resample)(
                    wave=wave[i], flux=flux[i], outwave=wave_new, ivar=ivar[i]
                )
                for i in range(len(flux))
            )
        )

    flux_new = np.array(flux_new)
    ivar_new = np.array(ivar_new)
    ## janky fix, find overlap and then fill
    # find overlap between new and old wavelength grid

    mask = flux_new == 0

    flux_new[mask] = fill_val
    ivar_new[mask] = fill_val
    return flux_new, ivar_new


def flux_conserving_resample(
    wave_new, wave, flux, ivar=None, fill_val=np.nan, verbose=False, n_workers=1
):
    """
    Resample a spectrum to a new wavelength grid using Flux conserving resampling.
    """
    if n_workers == 1:
        flux_new, ivar_new = zip(
            *[
                resample_flux(xout=wave_new, x=wave[i], flux=flux[i], ivar=ivar[i])
                for i in range(len(flux))
            ]
        )
    else:
        flux_new, ivar_new = zip(
            *Parallel(n_jobs=n_workers)(
                delayed(resample_flux)(
                    xout=wave_new, x=wave[i], flux=flux[i], ivar=ivar[i]
                )
                for i in range(len(flux))
            )
        )

    flux_new = np.array(flux_new)
    ivar_new = np.array(ivar_new)
    ## janky fix, find overlap and then fill
    mask = flux_new == 0

    flux_new[mask] = fill_val
    ivar_new[mask] = fill_val
    return flux_new, ivar_new


def spectroperfection_resample(
    wave_new, wave, flux, ivar=None, fill_val=np.nan, verbose=False
):
    """
    Resample a spectrum to a new wavelength grid using the spectroperf algorithm.
    """
    raise NotImplementedError("Not implemented yet")


def resample(
    wave_new,
    wave,
    flux,
    ivar=None,
    R=None,
    fill_val=np.nan,
    method="linear",
    n_workers=-1,
):
    """Resample a spectrum to a new wavelength grid.

    Parameters
    ----------
    wave_new : array-like
        New grid to resample to.
    wave :  array-like
        Original wavelength grid.
    flux :  array-like
        Original flux grid.
    ivar : array-like, optional
        Original ivar grid, by default None
    R : _type_, optional
        _description_, by default None
    fill_val : np.float, optional
        fill value for wavelength bins outside initial range, by default np.nan
    method : str, optional
        Method used to resample the spectra, choose out of
        "linear", "sn-cons", "flux-cons",  by default "linear"
    n_workers : int, optional
        Number of CPU threads to use, by default -1

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # set the number of parallel workers
    if n_workers <= 0:
        n_workers = multiprocessing.cpu_count()
    else:
        n_workers = min(int(n_workers), multiprocessing.cpu_count())

    if method not in ["linear", "sn-cons", "flux-cons", "spectroperfection"]:
        raise ValueError(f"Unknown resampling method: {method}")
    if method == "linear":
        return linear_resample(
            wave_new, wave, flux, ivar, fill_val=np.nan, n_workers=n_workers
        )
    elif method == "sn-cons":
        return sn_conserving_resample(
            wave_new, wave, flux, ivar, fill_val=np.nan, n_workers=n_workers
        )
    elif method == "flux-cons":
        return flux_conserving_resample(
            wave_new, wave, flux, ivar, fill_val=np.nan, n_workers=n_workers
        )
    elif method == "spectroperfection":
        return spectroperfection_resample(wave_new, wave, flux, ivar, fill_val=np.nan)
