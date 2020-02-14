import numpy as np
from astropy.nddata import Cutout2D
from astropy.modeling.fitting import LevMarLSQFitter
from photutils.psf.groupstars import DAOGroup
from astropy.table import Table, vstack

__all__ = ["dao_nstar_clamp", "dao_weight_map", "dao_nstar",
           "daophot_concat"]


def dao_nstar_clamp(p_old, p_new_raw, p_clamp):
    ''' The "clamp" for NSTAR routine
    Note
    ----
    StetsonPB 1987, PASP, 99, 191, p.208
    '''
    dp_raw = p_new_raw - p_old
    p_new = p_old + dp_raw / (1 + np.abs(dp_raw) / p_clamp)
    return p_new


def dao_weight_map(data, position, r_fit):
    ''' The weight for centering routine
    Note
    ----
    StetsonPB 1987, PASP, 99, 191, p.207
    https://iraf.net/irafhelp.php?val=daopars&help=Help+Page
    '''
    x0, y0 = position
    is_cut = False
    if np.any(np.array(data.shape) > (2 * r_fit + 1)):  # To save CPU
        is_cut = True
        cut = Cutout2D(data=data, position=(x0, y0),
                       size=(2 * r_fit + 1), mode='partial')
        data = cut.data
        x0, y0 = cut.to_cutout_position((x0, y0))

    nx, ny = data.shape[1], data.shape[0]
    yy_data, xx_data = np.mgrid[:ny, :nx]

    # add 1.e-6 to avoid zero division
    distance_map = np.sqrt((xx_data - x0)**2 + (yy_data - y0)**2) + 1.e-6
    dist = np.ma.array(data=distance_map, mask=(distance_map > r_fit))
    rsq = dist**2 / r_fit**2
    weight_map = 5.0 / (5.0 + rsq / (1.0 - rsq))
    return weight_map


def dao_nstar(data, psf, position=None, r_fit=2, flux_init=1, sky=0, err=None,
              fitter=LevMarLSQFitter(), full=True):
    '''
    psf: photutils.psf.FittableImageModel
    '''
    if position is None:
        position = ((data.shape[1] - 1) / 2, (data.shape[0] - 1) / 2)

    if err is None:
        err = np.zeros_like(data)

    psf_init = psf.copy()
    psf_init.flux = flux_init

    fbox = 2 * r_fit + 1  # fitting box size
    fcut = Cutout2D(data, position=position, size=fbox,
                    mode='partial')  # "fitting" cut
    fcut_err = Cutout2D(err, position=position, size=fbox, mode='partial').data
    fcut_skysub = fcut.data - sky  # Must be sky subtracted before PSF fitting
    pos_fcut_init = fcut.to_cutout_position(position)  # Order of x, y
    psf_init.x_0, psf_init.y_0 = fcut.center_cutout

    dao_weight = dao_weight_map(fcut_skysub, pos_fcut_init, r_fit)
    # astropy gets ``weight`` = 1 / sigma.. strange..
    astropy_weight = np.sqrt(dao_weight.data) / fcut_err
    astropy_weight[dao_weight.mask] = 0

    yy_fit, xx_fit = np.mgrid[:fcut_skysub.shape[1], :fcut_skysub.shape[0]]
    fit = fitter(psf_init, xx_fit, yy_fit, fcut_skysub, weights=dao_weight)
    pos_fit = fcut.to_original_position((fit.x_0, fit.y_0))
    fit.x_0, fit.y_0 = pos_fit

    if full:
        return (fit, pos_fit, fitter,
                astropy_weight, fcut, fcut_skysub, fcut_err)

    else:
        return fit, pos_fit, fitter


def dao_substar(data, position, fitted_psf, size):
    pass


def daophot_concat(filelist, crit_separation, xcol="x", ycol="y",
                   table_reader=Table.read, reader_kwargs={}):
    ''' Concatenates the DAOPHOT-like results
    filelist : list of path-like
        The list of file paths to be concatenated.
    '''

    tablist = []
    for fpath in filelist:
        tab = table_reader(fpath, **reader_kwargs)
        tablist.append(tab)
    tabs = vstack(tablist)
    if "group_id" in tabs.colnames:
        tabs.remove_column("group_id")
    tabs["id"] = np.arange(len(tabs)) + 1

    tabs[xcol].name = "x_0"
    tabs[ycol].name = "y_0"
    tabs_g = DAOGroup(crit_separation=crit_separation)(tabs)
    tabs_g["x_0"].name = xcol
    tabs_g["y_0"].name = ycol
    return tabs_g
