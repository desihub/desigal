import numpy as np


def _deredshift(data_in, z_in, z_out, exponent):
    """Implements the basic redshift equation"""
    data_out = np.atleast_2d(data_in)
    return data_out * (((1 + z_out) / (1 + z_in)) ** exponent)


def deredshift(data_in, z_in, z_out, data_type):
    """Redshift Correction for input data

    Parameters
    ----------
    data_in : numpy.ndarray or dict.
        Input data which is either flux values, wavelengths or ivars.
        They can be a single numpy array (i.e. camera coadded) or a dict with each camera.
        Default DESI units are assumed.
    z_in : float or numpy.ndarray
        input redshifts
    z_out : float or numpy.ndarray
        output redshifts
    data_type : str
        "flux", "wave" or "ivar"

    Returns
    -------
    numpy.ndarray
        redshift corrected value corresponding to data type
    """

    exponent_dict = {"flux": -1, "wave": 1, "ivar": 2}
    assert data_type in exponent_dict.keys(), "Not a valid Data Type"

    if z_in.ndim == 1:
        z_in = z_in[:, np.newaxis]
    exponent = exponent_dict[data_type]

    if isinstance(data_in, np.ndarray):
        return _deredshift(data_in, z_in, z_out, exponent)
    if isinstance(data_in, dict):
        data_out = {}
        for key in data_in.keys():
            data_out[key] = _deredshift(data_in[key], z_in, z_out, exponent)
        return data_out
    else:
        raise ValueError("Input data not of a valid type")
