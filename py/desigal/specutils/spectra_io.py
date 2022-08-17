import os
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from astropy.table import Table
from desispec.zcatalog import find_primary_spectra
import desispec.io
import desispec.spectra


def get_spectra(targetids, release, n_workers=-1, **kwargs):
    """
    Get spectra for a list of targetids.
    Uses desispec.zcatalog.find_primary_spectra to find the primary spectra for each targetid.
    Use kwargs to pass to desispec.io.read_spectra, else uses default values.

    Parameters
    ----------
    targetids : list
        List of targetids to get spectra for.
    release : str
        Data release to get spectra for.
    n_workers : int, optional
        Number of parallel threads to read the files, by default -1, i.e. all available threads.

    Returns
    -------
    desispec.spectra.Spectra
        Spectra for the targetids.
    """
    if n_workers <= 0:
        n_workers = multiprocessing.cpu_count()
    else:
        n_workers = min(int(n_workers), multiprocessing.cpu_count())
    targetids = np.array(targetids)
    spectro_redux_path = Path(os.environ["DESI_SPECTRO_REDUX"])
    release_path = spectro_redux_path / release

    # Replace this step by database call once that is available
    all_data = Table.read(release_path / "zcatalog" / f"zall-pix-{release}.fits")
    spectro_redux_path = Path(os.environ["DESI_SPECTRO_REDUX"])
    release_path = spectro_redux_path / release
    all_data = Table.read(release_path / "zcatalog" / f"zall-pix-{release}.fits")

    select_mask = np.isin(all_data["TARGETID"].value, targetids)
    sel_data = all_data[select_mask]
    del all_data

    sel_data["ZCAT_NSPEC"] = 0
    sel_data["ZCAT_PRIMARY"] = 0

    nspec, specprim = find_primary_spectra(sel_data, **kwargs)
    sel_data["ZCAT_NSPEC"] = nspec  # number of spectra for this object in catalog
    sel_data[
        "ZCAT_PRIMARY"
    ] = specprim  # True/False if this is the primary spectrum in catalog

    sel_data = sel_data[sel_data["ZCAT_PRIMARY"]]

    found_targets_bool = np.isin(targetids, sel_data["TARGETID"])

    if found_targets_bool.sum() < len(targetids):
        print(f"Target ids {targetids[~found_targets_bool]} not found!")

    if found_targets_bool.sum() > len(targetids):
        raise SystemExit("Unresolved duplicate targets present!")

    sel_spectra = Parallel(n_jobs=n_workers)(
        delayed(_read_spectra)(survey, program, healpix, targetid, release_path)
        for survey, program, healpix, targetid in zip(
            sel_data["SURVEY"],
            sel_data["PROGRAM"],
            sel_data["HEALPIX"],
            sel_data["TARGETID"],
        )
    )
    return desispec.spectra.stack(sel_spectra)


def _read_spectra(survey, program, healpix, targetid, release_path):
    """Read a single spectra file. Helper function of get_spectra."""
    data_path = (
        release_path
        / "healpix"
        / survey
        / program
        / str(int(healpix / 100))
        / str(healpix)
        / f"coadd-{survey}-{program}-{healpix}.fits"
    )
    spectra = desispec.io.read_spectra(data_path)
    mask = np.isin(spectra.fibermap["TARGETID"], targetid)
    spectra = spectra[mask]
    return spectra