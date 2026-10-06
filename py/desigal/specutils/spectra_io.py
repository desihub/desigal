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
import specprodDB.load as db


#: Directory under a release that holds the healpix-grouped coadds. This was
#: ``healpix`` up to and including loa, and became ``spectra`` in matterhorn.
COADD_DIRNAMES = ("healpix", "spectra")

#: Name of the healpix column in a zcatalog. Renamed from HEALPIX to UNIQPIX
#: in matterhorn; the values index the directory tree identically.
HEALPIX_COLUMNS = ("HEALPIX", "UNIQPIX")

#: The only zcatalog columns get_spectra needs. These files run to tens of GB
#: across 130+ columns, so reading all of them is both slow and liable to
#: exhaust memory: loa's zall-pix is 47 GB, of which these five are 3.9 GB.
ZCAT_COLUMNS = ("TARGETID", "SURVEY", "PROGRAM", "ZCAT_PRIMARY")

#: HDUs skipped when reading a coadd file. RESOLUTION is by far the largest and
#: nothing in the stacking path uses it. MASK is deliberately *not* skipped:
#: masks are not applied when coadding across cameras, so dropping them gives
#: quietly wrong spectra.
DEFAULT_SKIP_HDUS = ("EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG", "RESOLUTION")


def _spectro_redux():
    """Root of the spectroscopic reductions, from $DESI_SPECTRO_REDUX."""
    try:
        return Path(os.environ["DESI_SPECTRO_REDUX"])
    except KeyError:
        raise KeyError(
            "$DESI_SPECTRO_REDUX is not set. Source the DESI environment "
            "first, e.g. "
            "`source /global/common/software/desi/desi_environment.sh`."
        ) from None


def _version_sort_key(name):
    """Sort key for a ``vN``/``vN.M`` zcatalog directory name."""
    return [int(part) for part in name[1:].split(".")]


def _is_version_dir(path):
    """True for a directory named like a zcatalog version, e.g. v1 or v1.1."""
    if not path.is_dir() or not path.name.startswith("v"):
        return False
    try:
        _version_sort_key(path.name)
    except ValueError:
        return False
    return True


def list_releases(spectro_redux=None, require_spectra=True):
    """List the DESI data releases available in this environment.

    Releases are discovered from the filesystem rather than hardcoded, so a
    future release is picked up with no code change. A directory counts as a
    release if it contains a ``zall-pix-<name>.fits`` catalog and, unless
    ``require_spectra`` is False, a tree of coadded spectra. $DESI_SPECTRO_REDUX
    also holds well over a hundred personal and test reduction directories,
    some of which have a ``zcatalog`` of their own; requiring the release-named
    catalog is what separates them from the real thing.

    Parameters
    ----------
    spectro_redux : str or pathlib.Path, optional
        Root to search. Defaults to $DESI_SPECTRO_REDUX.
    require_spectra : bool, optional
        If True (the default) only return releases whose coadds are actually
        on disk. Some releases, jura at the time of writing, keep a zcatalog
        at NERSC but no spectra, so `get_spectra` cannot read from them.

    Returns
    -------
    list of str
        Release names, sorted alphabetically.
    """
    root = Path(spectro_redux) if spectro_redux is not None else _spectro_redux()
    releases = []
    for entry in sorted(root.iterdir()):
        try:
            if not entry.is_dir() or not (entry / "zcatalog").is_dir():
                continue
            if require_spectra and not any(
                (entry / name).is_dir() for name in COADD_DIRNAMES
            ):
                continue
            _zcatalog_path(entry.name, entry)
        except (FileNotFoundError, PermissionError, OSError):
            # Not a release, or a directory we cannot read into.
            continue
        releases.append(entry.name)
    return releases


def _coadd_dir(release_path):
    """Directory holding the healpix-grouped coadds for a release."""
    for name in COADD_DIRNAMES:
        candidate = release_path / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No coadded spectra found for release '{release_path.name}': none of "
        f"{list(COADD_DIRNAMES)} exist under {release_path}. Some releases "
        "(jura, at the time of writing) keep a zcatalog at NERSC but no "
        "spectra, in which case targets can be looked up but not read. "
        f"Releases that do have spectra: {list_releases()}"
    )


def _zcatalog_path(release, release_path):
    """Locate the ``zall-pix`` catalog for a release.

    The layout has moved three times: directly under ``zcatalog`` (fuji), in a
    version subdirectory (guadalupe through loa), and in a ``zall``
    subdirectory of that (matterhorn). Only those exact locations are checked,
    which also avoids the stale copies under ``zcatalog/v*/deprecated``.
    """
    zcatalog_dir = release_path / "zcatalog"
    if not zcatalog_dir.is_dir():
        raise FileNotFoundError(
            f"No zcatalog directory for release '{release}' at {zcatalog_dir}. "
            f"Available releases: {list_releases()}"
        )

    filename = f"zall-pix-{release}.fits"
    candidates = [zcatalog_dir / filename]
    for version_dir in sorted(
        (d for d in zcatalog_dir.iterdir() if _is_version_dir(d)),
        key=lambda d: _version_sort_key(d.name),
        reverse=True,
    ):
        candidates.append(version_dir / filename)
        candidates.append(version_dir / "zall" / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {filename} found for release '{release}'. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def _healpix_column(colnames):
    """Name of the healpix column present in a zcatalog."""
    for name in HEALPIX_COLUMNS:
        if name in colnames:
            return name
    raise KeyError(
        "zcatalog has none of the expected healpix columns "
        f"{list(HEALPIX_COLUMNS)}; found {list(colnames)[:20]}..."
    )


def _decode_bytes(table):
    """Decode any bytes columns of a pandas frame to str, in place."""
    for column in table.columns:
        values = table[column]
        if values.dtype == object and len(values) and isinstance(
            values.iloc[0], bytes
        ):
            table[column] = values.str.decode("utf-8")
    return table


def _release_has_db(release):
    """Whether the redshift database can actually serve this release.

    A schema may be absent (jura, kibo) or present but unpopulated
    (matterhorn, whose ``zpix`` table exists and is empty), so this checks for
    a row rather than just for the table.
    """
    try:
        db.setup_db(
            schema=release, hostname="specprod-db.desi.lbl.gov", username="desi"
        )
        return db.dbSession.query(db.Zpix.targetid).limit(1).count() > 0
    except Exception:
        return False


def get_spectra(
    targetids, release, n_workers=-1, use_db=True, zcat_table=None,
    skip_hdus=None, **kwargs
):
    """
    Get spectra for a list of targetids.
    Uses desispec.zcatalog.find_primary_spectra to find the primary spectra for each targetid.
    Use kwargs to pass to desispec.io.read_spectra, else uses default values.

    Parameters
    ----------
    targetids : list
        List of targetids to get spectra for.
    release : str
        Data release to get spectra for, e.g. "iron" or "loa". Call
        `list_releases` for the ones available in this environment.
    n_workers : int, optional
        Number of parallel threads to read the files, by default -1, i.e. all available threads.
    use_db : bool, optional
        Use the desi redshift database to get the list of spectra files, by default True.
        Needs an initial setup of the `~/.pgpass` file. See https://desi.lbl.gov/trac/wiki/DESIProductionDatabase#Setuppgpass
        Releases the database cannot serve fall back to the zcatalog FITS
        file with a warning; see Notes.
    zcat_table : astropy.table.Table, optional
        Pre-loaded redshift catalog to locate the spectra from, used only when
        ``use_db=False``. Useful for a custom or filtered catalog, or to avoid
        re-reading a large zcatalog across repeated calls.

        It must carry these columns:

        ``TARGETID``
            Target identifiers to match against.
        ``SURVEY``, ``PROGRAM``
            Used to build the coadd file path.
        ``HEALPIX`` or ``UNIQPIX``
            Healpix number, also part of the path. Named ``UNIQPIX`` from
            matterhorn on; either is accepted.
        ``ZCAT_PRIMARY``
            Optional. If absent it is computed with
            `desispec.zcatalog.find_primary_spectra`, which additionally
            needs ``ZWARN`` and the sort column (``TSNR2_LRG`` by default,
            override with ``sort_column=...``).

        Only rows with ``ZCAT_PRIMARY`` true are used, so a target present
        solely as a non-primary spectrum will be reported as not found.
    skip_hdus : tuple of str, optional
        HDUs not to read from each coadd file, passed to
        `desispec.io.read_spectra`. Defaults to `DEFAULT_SKIP_HDUS`, which
        drops EXP_FIBERMAP, SCORES, EXTRA_CATALOG and RESOLUTION. Pass an
        empty tuple to read everything. Note that skipping MASK is a bad
        idea: masks are not applied when coadding across cameras, so without
        them the spectra are quietly wrong.

    Returns
    -------
    desispec.spectra.Spectra
        Spectra for the targetids.

    Raises
    ------
    FileNotFoundError
        If the release does not exist, or has no coadded spectra on disk. The
        latter is the case for jura, which keeps a zcatalog at NERSC but no
        spectra.

    Notes
    -----
    Release layouts have changed over time and are handled transparently:

    * coadds live under ``healpix/`` up to loa and ``spectra/`` from
      matterhorn on;
    * the ``zall-pix`` catalog sits directly in ``zcatalog/`` (fuji), in a
      version subdirectory (guadalupe through loa), or in a ``zall``
      subdirectory of that (matterhorn), with the highest version winning;
    * the healpix column is named ``HEALPIX`` up to loa and ``UNIQPIX`` in
      matterhorn.

    Not every release is in the redshift database. At the time of writing
    jura and kibo have no ``zpix`` table and matterhorn's is empty, so those
    fall back to reading the zcatalog. The fallback is much slower -- for loa,
    about 1.5 s via the database against about 78 s via FITS -- so prefer the
    database where it is available.
    """
    if n_workers <= 0:
        n_workers = multiprocessing.cpu_count()
    else:
        n_workers = min(int(n_workers), multiprocessing.cpu_count())
    targetids = list(targetids)
    spectro_redux_path = _spectro_redux()
    release_path = spectro_redux_path / release
    if not release_path.is_dir():
        raise FileNotFoundError(
            f"No such release '{release}' under {spectro_redux_path}. "
            f"Available releases: {list_releases()}"
        )
    # Raises with an actionable message for a catalog-only release such as
    # jura, before any time is spent reading a multi-GB zcatalog.
    coadd_dir = _coadd_dir(release_path)

    if use_db and zcat_table is None and not _release_has_db(release):
        log = get_logger()
        log.warning(
            "The redshift database has no usable data for release '%s', so "
            "falling back to the zcatalog FITS file. This is slower; pass "
            "use_db=False to select it explicitly and silence this warning.",
            release,
        )
        use_db = False

    if use_db:
        sel_data = _sel_objects_db(release, targetids)
    elif zcat_table is not None:
        sel_data = _sel_objects_table(zcat_table, targetids)
    else:
        sel_data = _sel_objects_fits(release, release_path, targetids)

    sel_data = sel_data.set_index("TARGETID", drop=False)

    # Check before the .loc below rather than after it: .loc raises its own
    # KeyError ("[...] not in index") for anything missing, which says nothing
    # about what went wrong or what to do next.
    found_targets_bool = np.isin(targetids, sel_data.index.values)
    if not np.all(found_targets_bool):
        missing = np.asarray(targetids)[~found_targets_bool]
        message = (
            f"{missing.size} of {len(targetids)} requested target ids were not "
            f"found in release '{release}': {missing.tolist()[:10]}"
            f"{' ...' if missing.size > 10 else ''}."
        )
        if use_db:
            message += (
                " These were looked up in the redshift database, which can be "
                "incomplete relative to the zcatalog -- some fuji targets are "
                "in zall-pix but absent from fuji.zpix, for instance. Try "
                "use_db=False to search the zcatalog file instead."
            )
        else:
            message += (
                " Note that only ZCAT_PRIMARY spectra are selected, so a "
                "target observed only as a non-primary will not be found."
            )
        raise ValueError(message)

    sel_data = sel_data.loc[targetids]
    sel_data = Table.from_pandas(sel_data)

    # Group the targets by the file they live in, so each coadd file is opened
    # once however many targets it holds. Reading one target at a time reopened
    # a ~450 MB file per target, which dominated the runtime (#24).
    groups = {}
    for survey, program, healpix, targetid in zip(
        sel_data["SURVEY"],
        sel_data["PROGRAM"],
        sel_data["HEALPIX"],
        sel_data["TARGETID"],
    ):
        key = (str(survey), str(program), int(healpix))
        groups.setdefault(key, []).append(int(targetid))
    groups = list(groups.items())

    if skip_hdus is None:
        skip_hdus = DEFAULT_SKIP_HDUS

    # adding special case so as to have the option to parallelize externally
    if n_workers == 1:
        sel_spectra = [
            _read_spectra(survey, program, healpix, tids, coadd_dir, skip_hdus)
            for (survey, program, healpix), tids in groups
        ]
    else:
        sel_spectra = Parallel(n_jobs=min(n_workers, len(groups)))(
            delayed(_read_spectra)(
                survey, program, healpix, tids, coadd_dir, skip_hdus
            )
            for (survey, program, healpix), tids in groups
        )

    spectra = stack(sel_spectra)
    # stack() returns them grouped by file; restore the requested order.
    position = {
        int(targetid): row
        for row, targetid in enumerate(spectra.fibermap["TARGETID"])
    }
    return spectra[[position[int(t)] for t in targetids]]


def _sel_objects_fits(release, release_path, targetids, **kwargs):
    """Select objects from the fits file. Helper function of get_spectra.

    Reads in two passes -- TARGETID to find the rows of interest, then just
    those rows of the handful of columns actually needed. A zall-pix catalog
    runs to tens of GB over 130+ columns (47 GB for loa), so reading it whole
    is impractical.
    """
    zcat_path = _zcatalog_path(release, release_path)

    with fitsio.FITS(str(zcat_path)) as hdus:
        extnames = [hdu.get_extname() for hdu in hdus]
        zcat = hdus["ZCATALOG"] if "ZCATALOG" in extnames else hdus[1]
        colnames = zcat.get_colnames()
        healpix_column = _healpix_column(colnames)

        # Pass 1: TARGETID alone, to locate the rows we want.
        rows = np.flatnonzero(np.isin(zcat["TARGETID"].read(), targetids))
        if rows.size == 0:
            raise ValueError(
                f"None of the {len(targetids)} requested target ids were "
                f"found in {zcat_path}."
            )

        # Pass 2: only the needed columns, only the matching rows.
        columns = [c for c in ZCAT_COLUMNS if c in colnames]
        columns.append(healpix_column)
        if "ZCAT_PRIMARY" not in colnames:
            # find_primary_spectra needs these to work out the primary itself.
            columns += [
                c for c in ("ZWARN", kwargs.get("sort_column", "TSNR2_LRG"))
                if c in colnames and c not in columns
            ]
        sel_data = Table(zcat.read(columns=columns, rows=rows))

    if healpix_column != "HEALPIX":
        # matterhorn renamed HEALPIX to UNIQPIX; the values are the same.
        sel_data.rename_column(healpix_column, "HEALPIX")

    if "ZCAT_PRIMARY" not in sel_data.colnames:
        sel_data["ZCAT_NSPEC"] = 0
        sel_data["ZCAT_PRIMARY"] = 0

        nspec, specprim = find_primary_spectra(sel_data, **kwargs)
        sel_data["ZCAT_NSPEC"] = nspec  # number of spectra for this object in catalog
        sel_data["ZCAT_PRIMARY"] = (
            specprim  # True/False if this is the primary spectrum in catalog
        )

    sel_data = sel_data[sel_data["ZCAT_PRIMARY"]]
    sel_data = sel_data[["SURVEY", "PROGRAM", "HEALPIX", "TARGETID"]].to_pandas()
    return _decode_bytes(sel_data)


def _sel_objects_table(table, targetids, **kwargs):
    """Select objects from the table. Helper function of get_spectra."""
    select_mask = np.isin(table["TARGETID"].value, targetids)
    sel_data = table[select_mask]

    healpix_column = _healpix_column(sel_data.colnames)
    if healpix_column != "HEALPIX":
        # matterhorn renamed HEALPIX to UNIQPIX; the values are the same.
        sel_data.rename_column(healpix_column, "HEALPIX")

    if "ZCAT_PRIMARY" not in sel_data.colnames:
        sel_data["ZCAT_NSPEC"] = 0
        sel_data["ZCAT_PRIMARY"] = 0

        nspec, specprim = find_primary_spectra(sel_data, **kwargs)
        sel_data["ZCAT_NSPEC"] = nspec  # number of spectra for this object in catalog
        sel_data["ZCAT_PRIMARY"] = (
            specprim  # True/False if this is the primary spectrum in catalog
        )

    sel_data = sel_data[sel_data["ZCAT_PRIMARY"]]
    sel_data = sel_data[["SURVEY", "PROGRAM", "HEALPIX", "TARGETID"]].to_pandas()
    return _decode_bytes(sel_data)


def _sel_objects_db(release, targetids, **kwargs):
    """Select objects from the database. Helper function of get_spectra."""
    pgpass_path = Path(Path.home() / ".pgpass")
    if not pgpass_path.is_file():
        raise SystemExit(
            """Database access requires a ~/.pgpass file.
            See https://desi.lbl.gov/trac/wiki/DESIProductionDatabase#Setuppgpass for one time setup instructions.
            Else use `use_db=False` to use the slower fits table based search.
            You may also try running this from the command line and then rerun this cell.
            `cat /global/common/software/desi/desi_public.pgpass >> ~/.pgpass; chmod 600 ~/.pgpass`"""
        )
    db.log = get_logger(DEBUG)
    postgresql = db.setup_db(
        schema=release, hostname="specprod-db.desi.lbl.gov", username="desi"
    )

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


def _read_spectra(survey, program, healpix, targetids, coadd_dir, skip_hdus=None):
    """Read every requested target from one coadd file.

    Helper function of get_spectra. ``coadd_dir`` is the resolved coadd root
    for the release -- ``healpix`` up to loa, ``spectra`` from matterhorn on --
    as returned by `_coadd_dir`.

    Takes a list of target ids rather than one, so a file holding several
    requested targets is opened once rather than once per target.
    """
    healpix = int(healpix)
    data_path = (
        coadd_dir
        / survey
        / program
        / str(healpix // 100)
        / str(healpix)
        / f"coadd-{survey}-{program}-{healpix}.fits"
    )
    return desispec.io.read_spectra(
        str(data_path),
        targetids=list(targetids),
        skip_hdus=DEFAULT_SKIP_HDUS if skip_hdus is None else skip_hdus,
    )


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
    hdus = fitsio.FITS(infile, mode="r")

    targetrow = np.argwhere(hdus["FIBERMAP"].read(columns="TARGETID") == targetid)[0][0]
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
                fmap = encode_table(
                    Table(hdus[h].read(rows=targetrow), copy=True).as_array()
                )
        elif name == "EXP_FIBERMAP":
            if read_hdu["EXP_FIBERMAP"]:
                expfmap = encode_table(
                    Table(hdus[h].read(rows=targetrow), copy=True).as_array()
                )
        elif name == "SCORES":
            if read_hdu["SCORES"]:
                scores = encode_table(
                    Table(hdus[h].read(rows=targetrow), copy=True).as_array()
                )
        elif name == "EXTRA_CATALOG":
            if read_hdu["EXTRA_CATALOG"]:
                extra_catalog = encode_table(
                    Table(hdus[h].read(rows=targetrow), copy=True).as_array()
                )
        else:
            # Find the band based on the name
            mat = re.match(r"(.*)_(.*)", name)
            if mat is None:
                raise RuntimeError(
                    "FITS extension name {} does not contain the band".format(name)
                )
            band = mat.group(1).lower()
            type = mat.group(2)
            if band not in bands:
                bands.append(band)
            if type == "WAVELENGTH":
                if wave is None:
                    wave = {}
                # - Note: keep original float64 resolution for wavelength
                wave[band] = native_endian(hdus[h].read())
            elif type == "FLUX":
                if flux is None:
                    flux = {}
                flux[band] = native_endian(
                    hdus[h][targetrow : targetrow + 1, :].astype(ftype)
                )
            elif type == "IVAR":
                if ivar is None:
                    ivar = {}
                ivar[band] = native_endian(
                    hdus[h][targetrow : targetrow + 1, :].astype(ftype)
                )
            elif type == "MASK" and read_hdu["MASK"]:
                if mask is None:
                    mask = {}
                mask[band] = native_endian(
                    hdus[h][targetrow : targetrow + 1, :].astype(np.uint32)
                )
            elif type == "RESOLUTION" and read_hdu["RESOLUTION"]:
                if res is None:
                    res = {}
                res[band] = native_endian(
                    hdus[h][targetrow : targetrow + 1, :, :].astype(ftype)
                )
            else:
                pass
    hdus.close()
    duration = time.time() - t0
    log.info(iotime.format("read", infile, duration))

    # Construct the Spectra object from the data.  If there are any
    # inconsistencies in the sizes of the arrays read from the file,
    # they will be caught by the constructor.
    spec = Spectra(
        bands,
        wave,
        flux,
        ivar,
        mask=mask,
        resolution_data=res,
        fibermap=fmap,
        exp_fibermap=expfmap,
        meta=meta,
        extra=extra,
        extra_catalog=extra_catalog,
        single=single,
        scores=scores,
    )
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
