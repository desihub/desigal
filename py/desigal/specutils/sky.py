import os
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import desispec.io
import desispec.fluxcalibration


def get_sky(
    spectra=None,
    fibermap=None,
    exp_fibermap=None,
    release="fuji",
    n_workers=-1,
):
    """Get equivalent sky spectra for a set of coadded spectra.

    Parameters
    ----------
    spectra : desispec.spectra.Spectra,
        coadded spectra as a desispec object, by default None.
    fibermap : astropy.Table, optional
        If spectra object is not provided alternatively,
        fibermap and exp_fibermap may be provided, by default None.
    exp_fibermap : astropy.Table, optional
        If spectra object is not provided alternatively,
        fibermap and exp_fibermap may be provided, by default None.
    release : str, optional
        Name of data release, by default "everest".
    n_workers : int, optional
        Number of CPUs to spread the IO on. Vaulues<=0 will use all available CPUs, by default -1.

    Returns
    -------
    tuple of dictionaries
        sky_flux, sky_mask values for all the spectra broken in terms of camera.

    """
    if (spectra is None) and (fibermap is None) and (exp_fibermap is None):
        raise ValueError("Either spectra or fibermap and exp_fibermap must be provided")

    if spectra is not None:
        exp_fibermap = spectra.exp_fibermap
        fibermap = spectra.fibermap

    if (fibermap is None) ^ (exp_fibermap is None):
        raise ValueError(
            "fibermap and exp_fibermap must be both provided or spectra should be provided"
        )
    if n_workers <= 0:
        n_workers = multiprocessing.cpu_count()
    else:
        n_workers = min(int(n_workers), multiprocessing.cpu_count())

    spectro_redux_path = Path(os.environ["DESI_SPECTRO_REDUX"])

    sky_flux = {"b": [], "r": [], "z": []}
    sky_mask = {"b": [], "r": [], "z": []}

    if n_workers == 1:
        for target in fibermap["TARGETID"]:
            sky_flux_target, sky_mask_target = _get_target_sky(
                target, exp_fibermap, spectro_redux_path, release
            )
            for camera in ["b", "r", "z"]:
                sky_flux[camera].append(sky_flux_target[camera])
                sky_mask[camera].append(sky_mask_target[camera])

    else:

        sky_flux_target, sky_mask_target = zip(
            *Parallel(n_jobs=n_workers)(
                delayed(_get_target_sky)(
                    target, exp_fibermap, spectro_redux_path, release
                )
                for target in fibermap["TARGETID"]
            )
        )

        for s in sky_flux_target:
            for k, v in s.items():
                sky_flux[k].extend(v)
        for s in sky_mask_target:
            for k, v in s.items():
                sky_mask[k].extend(v)

    sky_flux = {key: np.array(value) for (key, value) in sky_flux.items()}
    sky_mask = {key: np.array(value) for (key, value) in sky_mask.items()}

    return sky_flux, sky_mask


def _preprocess_sky_frame(
    night, exp, petal, fiber, camera, spectro_redux_path, release, **kwargs
):
    exp_path = (
        spectro_redux_path / release / "exposures" / str(night) / str(exp).zfill(8)
    )
    sky_path = exp_path / f"sky-{camera}{petal}-{str(exp).zfill(8)}.fits"
    calib_path = exp_path / f"fluxcalib-{camera}{petal}-{str(exp).zfill(8)}.fits"
    cframe_path = exp_path / f"cframe-{camera}{petal}-{str(exp).zfill(8)}.fits"

    sky_data = desispec.io.read_sky(str(sky_path))
    sky_data.fibermap = None  # Need to set this for flux calibration function to work
    calib_data = desispec.io.read_flux_calibration(str(calib_path))
    cframe_data = desispec.io.read_frame(str(cframe_path), skip_resolution=True)

    sky_idx = fiber % 500  # find the array idx of the fiber

    # The following uses a for loop internally for a part.
    # To reduce compute we may want to select specific objects here rather than calibratng the whole
    desispec.fluxcalibration.apply_flux_calibration(
        sky_data, calib_data
    )  # modifies sky_data.flux and sky_data.ivar

    exp_sky_flux = sky_data.flux[sky_idx]
    exp_ivar = cframe_data.ivar[sky_idx]
    exp_sky_mask = np.logical_and(sky_data.mask[sky_idx], cframe_data.mask[sky_idx])

    return exp_sky_flux, exp_ivar, exp_sky_mask


def _get_target_sky(target, exp_fibermap, spectro_redux_path, release, **kwargs):
    sky_flux_target = {"b": [], "r": [], "z": []}
    sky_mask_target = {"b": [], "r": [], "z": []}

    target_table = exp_fibermap[exp_fibermap["TARGETID"] == target]
    for camera in ["b", "r", "z"]:
        cam_sky_flux = []
        cam_sky_ivar = []
        cam_sky_mask = []
        for night, exp, petal, fiber in zip(
            target_table["NIGHT"],
            target_table["EXPID"],
            target_table["PETAL_LOC"],
            target_table["FIBER"],
        ):
            exp_sky_flux, exp_sky_ivar, exp_sky_mask = _preprocess_sky_frame(
                night, exp, petal, fiber, camera, spectro_redux_path, release, **kwargs
            )
            cam_sky_flux.append(exp_sky_flux)
            cam_sky_ivar.append(exp_sky_ivar)
            cam_sky_mask.append(exp_sky_mask)

        cam_sky_flux = np.array(cam_sky_flux)
        cam_sky_ivar = np.array(cam_sky_ivar)
        cam_sky_mask = np.array(cam_sky_mask)
        sky_flux_target[camera].append(
            np.sum(cam_sky_flux * cam_sky_ivar, axis=0) / np.sum(cam_sky_ivar, axis=0)
        )
        sky_mask_target[camera].append(1 * np.all(cam_sky_ivar, axis=0))

    return sky_flux_target, sky_mask_target
