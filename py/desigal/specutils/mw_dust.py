import numpy as np
from desiutil.dust import SFDMap, dust_transmission
sfdmap = SFDMap()

def _mw_dust_correct(data_in, wave, ra, dec, exponent):
    """calculates the wl dependent MW dust correction"""
    ebv_sfd = sfdmap.ebv(ra, dec)
    transmission = np.array(
        [
            dust_transmission(wav, ebv)
            for wav, ebv in zip(np.atleast_2d(wave), np.atleast_1d(ebv_sfd))
        ]
    )
    data_out = np.atleast_2d(data_in) * transmission**exponent
    return data_out

def mw_dust_correct(data_in, wave, ra, dec, data_type):
    """MW dust correction for input data

    Parameters
    ----------
    data_in : numpy.ndarray or dict.
        Input data which is either flux values or ivars.
        They can be a single numpy array (i.e. camera coadded) or a dict with each camera.
        Default DESI units are assumed.
    wave : numpy.ndarray or dict
        input wavelength values
    ra : float or numpy.ndarray
        input right ascension
    dec : float of numpy.ndarray
        input declination
    data_type : str
        "flux" or "ivar"

    Returns
    -------
    numpy.ndarray
        MW dust corrected value corresponding to data type
    """

    exponent_dict = {"flux": -1, "ivar": 2}
    assert data_type in exponent_dict.keys(), "Not a valid Data Type"

    exponent = exponent_dict[data_type]

    if isinstance(data_in, np.ndarray):
        return _mw_dust_correct(data_in, wave, ra, dec, exponent)
    if isinstance(data_in, dict):
        data_out = {}
        for key in data_in.keys():
            data_out[key] = _mw_dust_correct(data_in[key], wave[key], ra, dec, exponent)
        return data_out
    else:
        raise ValueError("Input data not of a valid type")