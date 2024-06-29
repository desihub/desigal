import os
import re
import time
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import fitsio
from astropy.io import fits
from astropy.table import Table

from desiutil.io import encode_table
from desiutil.log import get_logger, DEBUG

import desispec.io
from desispec.io.util import native_endian, checkgzip
from desispec.io import iotime
from desispec.io.util import native_endian, checkgzip
from desispec.io import iotime
from desispec.zcatalog import find_primary_spectra
from desispec.spectra import Spectra, stack
import desispec.database.redshift as db


def get_spectra(targetids, release, n_workers=-1, use_db=True, zcat_table=None, **kwargs):
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
    zcat_table : astropy.Table.table, optional
        Use pre-loaded zcat table to get the list of spectra files. This is only used when 
        use_dp=False and a zcat_table is specified.
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
    elif (zcat_table is not None):
        sel_data = _sel_objects_table(zcat_table, targetids)
    else:
        sel_data = _sel_objects_fits(release, release_path, targetids)

    sel_data = sel_data.set_index("TARGETID", drop=False)
    sel_data = sel_data.loc[targetids]
    sel_data = Table.from_pandas(sel_data)
    file_sorted = sel_data.argsort(keys=["SURVEY","PROGRAM","HEALPIX","TARGETID"])
    inverse_sorted = np.argsort(file_sorted)
    sel_data = sel_data[file_sorted]
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
    sorted_spectra=[]
    for i in range(len(inverse_sorted)):
        sorted_spectra.append(sel_spectra[inverse_sorted[i]])
    return stack(sorted_spectra) #stack(np.array(sel_spectra)[inverse_sorted])


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
        if dtype == object:  # Only process object columns.
            # decode, or return original value if decode return Nan
            sel_data[col] = sel_data[col].str.decode("utf-8")

    return sel_data

def _sel_objects_table(table, targetids, **kwargs):
    """Select objects from the table. Helper function of get_spectra."""
    select_mask = np.isin(table["TARGETID"].value, targetids)
    sel_data = table[select_mask]

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
        if dtype == object:  # Only process object columns.
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
    print(data_path)
    spectra = read_single_spectrum(
        data_path, 
        targetid, 
        read_hdu={
            "FIBERMAP": True,
            "EXP_FIBERMAP": False,
            "SCORES": False,
            "EXTRA_CATALOG": False,
            "MASK": False,
            "RESOLUTION": False,
        }
    )
    #spectra = desispec.io.read_spectra(data_path)
    #mask = np.isin(spectra.fibermap["TARGETID"], targetid)
    #spectra = spectra[mask]
    return spectra


def read_single_spectrum(
    infile,
    targetid,
    single=False,
    read_hdu={
        "FIBERMAP": False,
        "EXP_FIBERMAP": False,
        "SCORES": False,
        "EXTRA_CATALOG": False,
        "MASK": False,
        "RESOLUTION": False,
        },
    ):    
    """
    Read single spectrum as Spectra object from FITS file.

    This reads data written by the write_spectra function.  A new Spectra
    object is instantiated and returned.

    Args:
        infile (str): path to read
        targetid (int): targetid of the spectrum to read
        single (bool): if True, keep spectra as single precision in memory.
        read_hdu (dict): Dict with hdu names as keys to skip or read hdu.

    Returns (Spectra):
        The object containing the data read from disk.

    """
    log = get_logger()
    infile = checkgzip(infile)
    ftype = np.float64
    if single:
        ftype = np.float32

    infile = os.path.abspath(infile)
    if not os.path.isfile(infile):
        raise IOError("{} is not a file".format(infile))

    t0 = time.time()
    hdus = fitsio.FITS(infile, mode='r')

    targetrow = np.argwhere(hdus["FIBERMAP"].read(columns="TARGETID")==targetid)[0][0]
    nhdu = len(hdus)

    # load the metadata.

    meta = dict(hdus[0].read_header())

    # initialize data objects

    bands = []
    fmap = None
    expfmap = None
    wave = None
    flux = None
    ivar = None
    mask = None
    res = None
    extra = None
    extra_catalog = None
    scores = None

    # For efficiency, go through the HDUs in disk-order.  Use the
    # extension name to determine where to put the data.  We don't
    # explicitly copy the data, since that will be done when constructing
    # the Spectra object.
            
    for h in range(1, nhdu):
        name = hdus[h].read_header()["EXTNAME"]
        if name == "FIBERMAP":
            if read_hdu["FIBERMAP"]:
                fmap = encode_table(Table(hdus[h].read(rows=targetrow), copy=True).as_array())
        elif name == "EXP_FIBERMAP":
            if read_hdu["EXP_FIBERMAP"]:
                expfmap = encode_table(Table(hdus[h].read(rows=targetrow), copy=True).as_array())
        elif name == "SCORES":
            if read_hdu["SCORES"]:
                scores = encode_table(Table(hdus[h].read(rows=targetrow), copy=True).as_array())
        elif name == "EXTRA_CATALOG":
            if read_hdu["EXTRA_CATALOG"]:
                extra_catalog = encode_table(Table(hdus[h].read(rows=targetrow), copy=True).as_array())
        else:
            # Find the band based on the name
            mat = re.match(r"(.*)_(.*)", name)
            if mat is None:
                raise RuntimeError("FITS extension name {} does not contain the band".format(name))
            band = mat.group(1).lower()
            type = mat.group(2)
            if band not in bands:
                bands.append(band)
            if type == "WAVELENGTH":
                if wave is None:
                    wave = {}
                #- Note: keep original float64 resolution for wavelength
                wave[band] = native_endian(hdus[h].read())
            elif type == "FLUX":
                if flux is None:
                    flux = {}
                flux[band] = native_endian(hdus[h][targetrow:targetrow+1, :].astype(ftype))
            elif type == "IVAR":
                if ivar is None:
                    ivar = {}
                ivar[band] = native_endian(hdus[h][targetrow:targetrow+1, :].astype(ftype))
            elif type == "MASK" and read_hdu["MASK"]:
                if mask is None:
                    mask = {}
                mask[band] = native_endian(hdus[h][targetrow:targetrow+1, :].astype(np.uint32))
            elif type == "RESOLUTION" and read_hdu["RESOLUTION"]:
                if res is None:
                    res = {}
                res[band] = native_endian(hdus[h][targetrow:targetrow+1, :, :].astype(ftype))
            else:
                pass
    hdus.close()
    duration = time.time() - t0
    log.info(iotime.format('read', infile, duration))

    # Construct the Spectra object from the data.  If there are any
    # inconsistencies in the sizes of the arrays read from the file,
    # they will be caught by the constructor.
    spec = Spectra(bands, wave, flux, ivar, mask=mask, resolution_data=res,
        fibermap=fmap, exp_fibermap=expfmap,
        meta=meta, extra=extra, extra_catalog=extra_catalog,
        single=single, scores=scores)
    return spec


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
