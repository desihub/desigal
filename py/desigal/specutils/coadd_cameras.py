import numpy as np


def coadd_cameras(flux_cam, wave_cam, ivar_cam, mask_cam=None):
    """Adds spectra from the three cameras as long as they have the same number of wavelength bins.
    This is not a replacement for desispec.coaddition.coadd_cameras,
    but a simpler (versatile and faster) implementation which uses only numpy.
    This also assumes the input spectra grid are already aligned
    (i.e. same wavelength grid in the overlapping regions),
    This is likely the case if the spectra are from the official data releases.

    Parameters
    ----------
    flux_cam : dict
        Dictionary containing the flux values from the three cameras
    wave_cam : dict
        Dictionary containing the wavelength values from the three cameras
    ivar_cam : dict
        Dictionary containing the inverse variance values from the three cameras
    mask_cam : dict, optional
        Dictionary containing the mask values from the three cameras

    Returns
    -------
    Tuple
        returns the combined flux, wavelength and inverse variance grids.
    """
    sbands = np.array(["b", "r", "z"])  # bands sorted by inc. wavelength
    # create wavelength array
    wave = None
    tolerance = 0.0001  # A , tolerance
    shifts = {}

    for b in sbands:
        wave_camera = np.atleast_2d(wave_cam[b].copy())
        if wave is None:
            wave = wave_camera
        else:
            shifts[b] = np.sum(
                np.all((wave + tolerance) < wave_camera[:, 0][:, None], axis=0)
            )
            wave = np.append(
                wave,
                wave_camera[
                    :, np.all(wave_camera > (wave[:, -1][:, None] + tolerance), axis=0)
                ],
                axis=1,
            )
    nwave = wave.shape[1]
    blue = sbands[0]
    ntarget = len(flux_cam[blue])
    flux = None
    ivar = None
    mask = None
    for b in sbands:
        flux_camera = np.atleast_2d(flux_cam[b].copy())
        ivar_camera = np.atleast_2d(ivar_cam[b].copy())
        ivar_camera[ivar_camera <= 0] = 0
        if mask_cam is not None:
            mask_camera = np.atleast_2d(mask_cam[b].astype(bool))
            ivar_camera[mask_camera] = 0
        if flux is None:
            flux = np.zeros((ntarget, nwave), dtype=flux_cam[blue].dtype)
            flux[:, : flux_camera.shape[1]] += flux_camera * ivar_camera
            ivar = np.zeros((ntarget, nwave), dtype=flux_cam[blue].dtype)
            ivar[:, : ivar_camera.shape[1]] += ivar_camera
            if mask is not None:
                mask = np.ones((ntarget, nwave), dtype=mask_cam[blue].dtype)
                mask[:, : mask_camera.shape[1]] &= mask_camera
        else:
            flux[:, shifts[b] : (shifts[b] + flux_camera.shape[1])] += (
                flux_camera * ivar_camera
            )
            ivar[:, shifts[b] : (shifts[b] + ivar_camera.shape[1])] += ivar_camera
            if mask is not None:
                mask[:, shifts[b] : (shifts[b] + mask_camera.shape[1])] &= mask_camera

    flux = flux / ivar
    flux[~np.isfinite(flux)] = 0
    ivar[~np.isfinite(ivar)] = 0

    if wave_cam[blue].ndim == 1:
        wave = np.squeeze(wave)
    if mask_cam is not None:
        return flux, wave, ivar, mask
    else:
        return flux, wave, ivar
