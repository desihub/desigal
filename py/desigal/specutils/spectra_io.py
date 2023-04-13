import os
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from desispec.zcatalog import find_primary_spectra
import desispec.io
import desispec.spectra
from desiutil.log import get_logger, DEBUG
import desispec.database.redshift as db


def get_spectra(targetids, release, n_workers=-1, use_db=True, **kwargs):
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
    use_db : bool, optional
        Use the desi redshift database to get the list of spectra files, by default True.
        Needs an initial setup of the `~/.pgpass` file. See https://desi.lbl.gov/trac/wiki/DESIProductionDatabase#Setuppgpass
    Returns
    -------
    desispec.spectra.Spectra
        Spectra for the targetids.
    """
    if n_workers <= 0:
        n_workers = multiprocessing.cpu_count()
    else:
        n_workers = min(int(n_workers), multiprocessing.cpu_count())
    targetids = list(targetids)
    spectro_redux_path = Path(os.environ["DESI_SPECTRO_REDUX"])
    release_path = spectro_redux_path / release

    if use_db:
        sel_data = _sel_objects_db(release, targetids)
    else:
        sel_data = _sel_objects_fits(release, release_path, targetids)

    sel_data = sel_data.set_index("TARGETID", drop=False)
    sel_data = sel_data.loc[targetids]
    found_targets_bool = np.isin(targetids, sel_data["TARGETID"])
    if ~np.all(found_targets_bool):
        raise ValueError(
            "Spectra for target ids {targetids[~found_targets_bool]} not found!"
        )

    # adding special case so as to have the option to parallelize externally
    if n_workers == 1:
        sel_spectra = [
            _read_spectra(survey, program, healpix, targetid, release_path)
            for survey, program, healpix, targetid in zip(
                sel_data["SURVEY"],
                sel_data["PROGRAM"],
                sel_data["HEALPIX"],
                sel_data["TARGETID"],
            )
        ]
    else:
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


def _sel_objects_fits(release, release_path, targetids, **kwargs):
    """Select objects from the fits file. Helper function of get_spectra."""
    # Replace this step by database call once that is available
    all_data = Table.read(
        release_path / "zcatalog" / f"zall-pix-{release}.fits",
        format="fits",
    )

    select_mask = np.isin(all_data["TARGETID"].value, targetids)
    sel_data = all_data[select_mask]
    del all_data

    if "ZCAT_PRIMARY" not in sel_data.colnames:
        sel_data["ZCAT_NSPEC"] = 0
        sel_data["ZCAT_PRIMARY"] = 0

        nspec, specprim = find_primary_spectra(sel_data, **kwargs)
        sel_data["ZCAT_NSPEC"] = nspec  # number of spectra for this object in catalog
        sel_data[
            "ZCAT_PRIMARY"
        ] = specprim  # True/False if this is the primary spectrum in catalog

    sel_data = sel_data[sel_data["ZCAT_PRIMARY"]]
    sel_data = sel_data[["SURVEY", "PROGRAM", "HEALPIX", "TARGETID"]].to_pandas()
    for col, dtype in sel_data.dtypes.items():
        if dtype == np.object:  # Only process object columns.
            # decode, or return original value if decode return Nan
            sel_data[col] = sel_data[col].str.decode("utf-8")

    return sel_data


def _sel_objects_db(release, targetids, **kwargs):
    """Select objects from the database. Helper function of get_spectra."""
    pgpass_path = Path(Path.home() / ".pgpass")
    if not pgpass_path.is_file():
        raise SystemExit(
            """Database access requires a ~/.pgpass file.
            See https://desi.lbl.gov/trac/wiki/DESIProductionDatabase#Setuppgpass for one time setup instructions.
            Else use use_db=False to use the slower fits table based search."""
        )
    db.log = get_logger(DEBUG)
    postgresql = db.setup_db(schema=release, hostname='specprod-db.desi.lbl.gov', username='desi')
    
    q = (
        db.dbSession.query(
            db.Zpix.survey,
            db.Zpix.program,
            db.Zpix.healpix,
            db.Zpix.targetid,
        )
        .filter(db.Zpix.targetid.in_(targetids))
        .filter(db.Zpix.zcat_primary == True)
        .all()
    )
    sel_data = pd.DataFrame(q, columns=["SURVEY", "PROGRAM", "HEALPIX", "TARGETID"])

    return sel_data


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


# def _read_spectra(survey, program, healpix, targetid, release_path):
#     """Read a single spectra file. Helper function of get_spectra."""
#     data_path = (
#         release_path
#         / "healpix"
#         / survey
#         / program
#         / str(int(healpix / 100))
#         / str(healpix)
#         / f"coadd-{survey}-{program}-{healpix}.fits"
#     )

#     hdus = fits.open(data_path, memap=True)
#     mask = np.isin(hdus[1].data["TARGETID"], targetid)
#     mask = np.where(mask)[0]

#     exp_mask = np.isin(hdus[2].data["TARGETID"], targetid)
#     exp_mask = np.where(exp_mask)[0]

#     spectra = desispec.spectra.Spectra()
#     # Doing this to avoid all the checks in the Spectra constructor
#     spectra.wave={"b": hdus[3].data.copy(), "r": hdus[8].data.copy(), "z": hdus[13].data.copy()}
#     spectra.flux={"b": hdus[4].data[mask].copy(),"r": hdus[9].data[mask].copy(),"z": hdus[14].data[mask].copy(),}
#     spectra.ivar={"b": hdus[5].data[mask].copy(),"r": hdus[10].data[mask].copy(),"z": hdus[15].data[mask].copy(),}
#     spectra.fibermap=hdus[1].data[mask].copy()
#     spectra.exp_fibermap=hdus[2].data[exp_mask].copy()
    ###TODO: ADD MASK and R
    

    return spectra

def write_binned_stacks(
    outfile,
    wave,
    flux,
    ivar,
    resolution=None,
    stackids=None,
    stack_redshift=None,
    table_column_dict={},
    table_format_dict={},
):
    """
    Save spectra to a fits file compatible with FastSpecFit stackfit

    Parameters
    ----------
    outfile : str
        Path and file name to save the fits file.
    wave : np.array
        1-Dimensional numpy array of the wavelength values of the spectra
    flux : np.array
        2-Dimensional numpy array of the flux values
    ivar : np.array
        1-Dimensional numpy array of the ivar values
    resolution : np.array
        3-Dimensional numpy array of the resolution matrices of the spectra. Saved in the same format as in desispec.resolution.Resolution: array[nspec, ndiag, nwave]
    stackids : np.array
        1-Dimensional numpy array of unique stackids. Default np.arange(nspec)
    stack_redshift : np.array
        1-Dimensional numpy array of the redshifts of the stacks. Default np.zeros(nspec)
    table_column_dict : dict
        Dictionary with column names(keys) and data to be included in the fits file.
    table_format_dict : dict
        Dictionary with column names(keys) and formats of the columns to be included in the fits file.
    Returns
    -------
    Nothing
        Saves spectra to fits file.
    """
    from astropy.io import fits

    nobj, _ = flux.shape
    if np.all(stackids == None):
        stackids = np.arange(nobj)
    if np.all(stack_redshift == None):
        stack_redshift = np.zeros(nobj)

    hdulist = []

    hdr = fits.Header()
    hdr[
        "COMMENT"
    ] = "Stack file created using desihub.desigal function write_binned_stacks"
    empty_primary = fits.PrimaryHDU(header=hdr)
    hdulist.append(empty_primary)

    hduflux = fits.ImageHDU(flux.astype("f4"))
    hduflux.header["EXTNAME"] = "FLUX"
    hdulist.append(hduflux)

    hduivar = fits.ImageHDU(ivar.astype("f4"))
    hduivar.header["EXTNAME"] = "IVAR"
    hdulist.append(hduivar)

    hduwave = fits.ImageHDU(wave.astype("f8"))
    hduwave.header["EXTNAME"] = "WAVE"
    hduwave.header["BUNIT"] = "Angstrom"
    hduwave.header["AIRORVAC"] = ("vac", "vacuum wavelengths")
    hdulist.append(hduwave)

    if ~np.all(resolution == None):
        hdures = fits.ImageHDU(resolution.astype("f4"))
        hdures.header["EXTNAME"] = "RES"
        hdulist.append(hdures)

    c1 = fits.Column(name="STACKID", array=stackids, format="K")
    c2 = fits.Column(name="Z", array=stack_redshift, format="D")
    columns = [c1, c2]
    for key in table_column_dict.keys():
        if table_format_dict[key][0] == "P":
            columns.append(
                fits.Column(
                    name=key,
                    array=np.array(table_column_dict[key], dtype="object"),
                    format=table_format_dict[key],
                )
            )
        else:
            columns.append(
                fits.Column(
                    name=key,
                    array=table_column_dict[key],
                    format=table_format_dict[key],
                )
            )

    hdutable = fits.BinTableHDU.from_columns(columns)
    hdutable.header["EXTNAME"] = "STACKINFO"
    hdulist.append(hdutable)

    hx = fits.HDUList(hdulist)

    hx.writeto(outfile, overwrite=True, checksum=True)
