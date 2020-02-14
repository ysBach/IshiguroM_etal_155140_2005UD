import numpy as np
from warnings import warn
from astropy.nddata import CCDData, Cutout2D
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.centroids import GaussianConst2D, centroid_com, fit_2dgaussian

from .util import Gaussian2D_correct
from .background import sky_fit

__all__ = ["find_center_2dg", "find_centroid", "find_centroid_com"]


def scaling_shift(pos_old, pos_new_naive, max_shift_step=None, verbose=False):
    dx = pos_new_naive[0] - pos_old[0]
    dy = pos_new_naive[1] - pos_old[1]
    shift = np.sqrt(dx**2 + dy**2)

    if max_shift_step is None:
        return dx, dy, shift

    if shift > max_shift_step:
        scale = max_shift_step / shift
        shift *= scale
        dx *= scale
        dy *= scale
        if verbose:
            print(f"shift({shift:.3f})"
                  + f" > max_shift_step({max_shift_step:.3f}). "
                  + f" shift truncated to {max_shift_step:.3f} pixels.")

    return dx, dy, shift


# TODO: Maybe make ssky to change as position updates.
def find_center_2dg(ccd, position_xy, cbox_size=5., csigma=3., ssky=0,
                    sky_annulus=None, sky_kw={},
                    maxiters=5, error=None, atol_shift=1.e-4, max_shift=1,
                    max_shift_step=None,
                    verbose=False, full=True, full_history=False):
    ''' Find the center of the image by 2D Gaussian fitting.

    Parameters
    ----------
    ccd : CCDData or ndarray
        The whole image which the `position_xy` is calculated.

    position_xy : array-like
        The position of the initial guess in image XY coordinate.

    cbox_size : float or int, optional
        The size of the box to find the centroid. Recommended to use 2.5
        to 4.0 times the seeing FWHM.  Minimally about 5 pixel is
        recommended. If extended source (e.g., comet), recommend larger
        cbox.
        See:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?centerpars

    csigma : float or int, optional
        The parameter to use in sigma-clipping. Using pixels only above
        3-simga level for centroiding is recommended. See Ma+2009,
        Optics Express, 17, 8525.

    ssky : float, optional
        The sample standard deviation of the sky or background. It will
        be used instead of ``sky_annulus`` if ``sky_annulus`` is
        ``None``. The pixels above the local minima (of the array of
        size ``cbox_size``) plus ``csigma*ssky`` will be used for
        centroid, following the default of IRAF's
        ``noao.digiphot.apphot``:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?centerpars.hlp

    sky_annulus : `~photutils.Aperture` annulus object
        The annulus to estimate the sky. All ``_shape_params`` of the
        object will be kept, while positions will be updated according
        to the new centroids. The initial input, therefore, does not
        need to have the position information (automatically initialized
        by ``position_xy``). If ``None`` (default), the constant
        ``ssky`` value will be used instead.

    sky_kw : dict, optional.
        The keyword arguments of `.backgroud.sky_fit`.

    tol_shift : float
        The absolute tolerance for the shift. If the shift in centroid
        after iteration is smaller than this, iteration stops.

    max_shift: float
        The maximum acceptable shift. If shift is larger than this,
        raises warning.

    max_shift_step : float, None, optional
        The maximum acceptable shift for each iteration. If the shift
        (say ``naive_shift``) is larger than this, the actual shift will
        be ``max_shift_step`` towards the direction identical to
        ``naive_shift``. If ``None`` (default), no such truncation is
        done.

    error : CCDData or ndarray
        The 1-sigma uncertainty map used for fitting.

    verbose : bool
        Whether to print how many iterations were needed for the
        centroiding.

    full : bool
        Whether to return the original and final cutout images.

    full_history : bool, optional
        Whether to return all the history of memory-heavy objects,
        including cutout ccd, cutout of the error frame, evaluated array
        of the fitted models. Most likely this is turned on only to
        check whether the centroiding process works correctly (i.e.,
        kind of debugging purpose).

    Returns
    -------
    newpos : tuple
        The iteratively found centroid position.

    shift : float
        Total shift from the initial position.

    fulldict : dict
        The ``dict`` when returned if ``full=True``:
            * ``positions``: `Nx2` numpy.array
        The history of ``x`` and ``y`` positions. The 0-th element is
        the initial position and the last element is the final fitted position.
    '''
    if sky_annulus is not None:
        import copy
        ANNULUS = copy.deepcopy(sky_annulus)
    else:
        ANNULUS = None

    def _center_2dg_iteration(ccd, position_xy, cbox_size=5., csigma=3.,
                              max_shift_step=None, ssky=0,
                              error=None, verbose=False):
        ''' Find the intensity-weighted centroid of the image iteratively

        Returns
        -------
        position_xy : float
            The centroided location in the original image coordinate in
            image XY.

        shift : float
            The total distance between the initial guess and the fitted
            centroid, i.e., the distance between `(xc_img, yc_img)` and
            `position_xy`.
        '''

        cut = Cutout2D(ccd.data, position=position_xy, size=cbox_size)
        e_cut = Cutout2D(error.data, position=position_xy, size=cbox_size)

        if ANNULUS is not None:
            ANNULUS.positions = position_xy
            ssky = sky_fit(ccd=ccd, annulus=ANNULUS, **sky_kw)["ssky"][0]

        cthresh = np.min(cut.data) + csigma * ssky
        mask = (cut.data < cthresh)
        # using pixels only above med + 3*std for centroiding is recommended.
        # See Ma+2009, Optics Express, 17, 8525
        # -- I doubt this... YPBach 2019-07-08 10:43:54 (KST: GMT+09:00)

        if verbose:
            n_all = np.size(mask)
            n_rej = np.count_nonzero(mask.astype(int))
            print(f"\t{n_rej} / {n_all} rejected [threshold = {cthresh:.3f} "
                  + f"from min ({np.min(cut.data):.3f}) "
                  + f"+ csigma ({csigma}) * ssky ({ssky:.3f})]")

        if ccd.mask is not None:
            cutmask = Cutout2D(ccd.mask, position=position_xy, size=cbox_size)
            mask += cutmask

        yy, xx = np.mgrid[:cut.data.shape[0], :cut.data.shape[1]]
        g_fit = fit_2dgaussian(data=cut.data,
                               error=e_cut.data,
                               mask=mask)
        g_fit = Gaussian2D_correct(g_fit)
        x_c_cut = g_fit.x_mean.value
        y_c_cut = g_fit.y_mean.value
        # The position is in the cutout image coordinate, e.g., (3, 3).

        pos_new_naive = cut.to_original_position((x_c_cut, y_c_cut))
        # convert the cutout image coordinate to original coordinate.
        # e.g., (3, 3) becomes something like (137, 189)

        dx, dy, shift = scaling_shift(position_xy, pos_new_naive,
                                      max_shift_step=max_shift_step,
                                      verbose=verbose)
        pos_new = np.array([position_xy[0] + dx, position_xy[1] + dy])

        return pos_new, shift, g_fit, cut, e_cut, g_fit(xx, yy)

    position_init = np.array(position_xy)
    if position_init.shape != (2,):
        raise TypeError("position_xy must have two and only two (xy) values.")

    if not isinstance(ccd, CCDData):
        _ccd = CCDData(ccd, unit='adu')  # Just a dummy unit
    else:
        _ccd = ccd.copy()

    if error is not None:
        if not isinstance(error, CCDData):
            _error = CCDData(error, unit='adu')  # Just a dummy unit
        else:
            _error = error.copy()
    else:
        _error = np.ones_like(_ccd.data)

    i_iter = 0
    positions = [position_init]
    d = 0

    if full:
        mods = []
        shift = []
        cuts = []
        e_cuts = []
        fits = []
        fit_params = {}
        for k in GaussianConst2D.param_names:
            fit_params[k] = []

    if verbose:
        print(f"Initial xy: {position_init} [0-indexing]\n"
              + f"\t(max iteration {maxiters:d}, "
              + f"shift tolerance {atol_shift} pixel)")

    for i_iter in range(maxiters):
        xy_old = positions[-1]

        if verbose:
            print(f"Iteration {i_iter+1:d} / {maxiters:d}: ")

        res = _center_2dg_iteration(ccd=_ccd,
                                    position_xy=xy_old,
                                    cbox_size=cbox_size,
                                    csigma=csigma,
                                    ssky=ssky,
                                    error=_error,
                                    max_shift_step=max_shift_step,
                                    verbose=verbose)
        newpos, d, g_fit, cut, e_cut, fit = res

        if d < atol_shift:
            if verbose:
                print(f"Finishing iteration (shift {d:.5f} < tol_shift).")
            break

        positions.append(newpos)

        if full:
            for k, v in zip(g_fit.param_names, g_fit.parameters):
                fit_params[k].append(v)
            shift.append(d)
            mods.append(g_fit)
            cuts.append(cut)
            e_cuts.append(e_cut)
            fits.append(fit)

        if verbose:
            print(f"\t({newpos[0]:.2f}, {newpos[1]:.2f}), shifted {d:.2f}")

    total_dx_dy = positions[-1] - positions[0]
    total_shift = np.sqrt(np.sum(total_dx_dy**2))

    if verbose:
        print(f"Final shift: dx={total_dx_dy[0]:+.2f}, "
              + f"dy={total_dx_dy[1]:+.2f}, "
              + f"total_shift={total_shift:.2f}")

    if total_shift > max_shift:
        warn(f"Object with initial position {position_xy} "
             + f"shifted larger than {max_shift} ({total_shift:.2f}).")

    if full:
        if full_history:
            fulldict = dict(positions=np.atleast_2d(positions),
                            shifts=np.atleast_1d(shift),
                            fit_models=mods,
                            fit_params=fit_params,
                            cuts=cuts,
                            e_cuts=e_cuts,
                            fits=fits)
        else:
            fulldict = dict(positions=np.atleast_2d(positions),
                            shifts=np.atleast_1d(shift),
                            fit_models=mods[-1],
                            fit_params=fit_params,
                            cuts=cuts[-1],
                            e_cuts=e_cuts[-1],
                            fits=fits[-1])
        return positions[-1], total_shift, fulldict

    return positions[-1], total_shift


# TODO: Add error-bar of the centroids by accepting error-map
def find_centroid(ccd, position_xy, centroider=centroid_com,
                  maxiters=5, cbox_size=5., csigma=3.,
                  ssky=0, tol_shift=1.e-4, max_shift=1,
                  max_shift_step=None, verbose=False, full=False):
    ''' Find the intensity-weighted centroid iteratively.
    Simply run `centroiding_iteration` function iteratively for
    `maxiters` times. Given the initial guess of centroid position in
    image xy coordinate, it finds the intensity-weighted centroid
    (center of mass) after rejecting pixels by sigma-clipping.

    Parameters
    ----------
    ccd : CCDData or ndarray
        The whole image which the `position_xy` is calculated.

    position_xy : array-like
        The position of the initial guess in image XY coordinate.

    cbox_size : float or int, optional
        The size of the box to find the centroid. Recommended to use 2.5
        to 4.0 times the seeing FWHM.  Minimally about 5 pixel is
        recommended. If extended source (e.g., comet), recommend larger
        cbox.
        See:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?centerpars

    csigma : float or int, optional
        The parameter to use in sigma-clipping. Using pixels only above
        3-simga level for centroiding is recommended. See Ma+2009,
        Optics Express, 17, 8525.
    ssky : float, optional
        The sample standard deviation of the sky or background. The
        pixels above the local minima (of the array of size
        ``cbox_size``) plus ``csigma*ssky`` will be used for centroid,
        following the default of IRAF's ``noao.digiphot.apphot``.
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?centerpars.hlp

    tol_shift : float
        The absolute tolerance for the shift. If the shift in centroid
        after iteration is smaller than this, iteration stops.

    max_shift: float
        The maximum acceptable shift. If shift is larger than this,
        raises warning.

    max_shift_step : float, None, optional
        The maximum acceptable shift for each iteration. If the shift
        (say ``naive_shift``) is larger than this, the actual shift will
        be ``max_shift_step`` towards the direction identical to
        ``naive_shift``. If ``None`` (default), no such truncation is
        done.

    verbose : bool
        Whether to print how many iterations were needed for the
        centroiding.

    full : bool
        Whether to return the original and final cutout images.

    Returns
    -------
    com_xy : list
        The iteratively found centroid position.
    '''

    def _centroiding_iteration(ccd, position_xy, centroider=centroid_com,
                               cbox_size=5., csigma=3.,
                               max_shift_step=None, ssky=0, verbose=False):
        ''' Find the intensity-weighted centroid of the image iteratively

        Returns
        -------
        position_xy : float
            The centroided location in the original image coordinate
            in image XY.

        shift : float
            The total distance between the initial guess and the
            fitted centroid, i.e., the distance between `(xc_img,
            yc_img)` and `position_xy`.
        '''

        x_init, y_init = position_xy
        cutccd = Cutout2D(ccd.data, position=position_xy, size=cbox_size)
        cthresh = np.min(cutccd.data) + csigma * ssky
        # using pixels only above med + 3*std for centroiding is recommended.
        # See Ma+2009, Optics Express, 17, 8525
        # -- I doubt this... YPBach 2019-07-08 10:43:54 (KST: GMT+09:00)

        mask = (cutccd.data < cthresh)

        if verbose:
            n_all = np.size(mask)
            n_rej = np.count_nonzero(mask.astype(int))
            print(f"\t{n_rej} / {n_all} rejected [threshold = {cthresh:.3f} "
                  + f"from min ({np.min(cutccd.data):.3f}) "
                  + f"+ csigma ({csigma}) * ssky ({ssky:.3f})]")

        if ccd.mask is not None:
            mask += ccd.mask

        x_c_cut, y_c_cut = centroider(data=cutccd.data, mask=mask)
        # The position is in the cutout image coordinate, e.g., (3, 3).

        x_c, y_c = cutccd.to_original_position((x_c_cut, y_c_cut))
        # convert the cutout image coordinate to original coordinate.
        # e.g., (3, 3) becomes something like (137, 189)

        pos_new_naive = cutccd.to_original_position((x_c_cut, y_c_cut))
        # convert the cutout image coordinate to original coordinate.
        # e.g., (3, 3) becomes something like (137, 189)

        dx, dy, shift = scaling_shift(position_xy, pos_new_naive,
                                      max_shift_step=max_shift_step,
                                      verbose=verbose)
        pos_new = (position_xy[0] + dx, position_xy[1] + dy)

        return pos_new, shift

    if not isinstance(ccd, CCDData):
        _ccd = CCDData(ccd, unit='adu')  # Just a dummy
    else:
        _ccd = ccd.copy()

    i_iter = 0
    xc_iter = [position_xy[0]]
    yc_iter = [position_xy[1]]
    shift = []
    d = 0
    if verbose:
        print(f"Initial xy: ({xc_iter[0]}, {yc_iter[0]}) [0-index]")
        print(f"\t(max iteration {maxiters:d}, shift tolerance {tol_shift})")

    for i_iter in range(maxiters):
        xy_old = (xc_iter[-1], yc_iter[-1])
        if verbose:
            print(f"Iteration {i_iter+1:d} / {maxiters:d}: ")
        (x, y), d = _centroiding_iteration(ccd=_ccd,
                                           position_xy=xy_old,
                                           cbox_size=cbox_size,
                                           csigma=csigma,
                                           ssky=ssky,
                                           max_shift_step=max_shift_step,
                                           verbose=verbose)
        xc_iter.append(x)
        yc_iter.append(y)
        shift.append(d)
        if d < tol_shift:
            if verbose:
                print(f"Finishing iteration (shift {d:.5f} < tol_shift).")
            break
        if verbose:
            print(f"\t({x:.2f}, {y:.2f}), shifted {d:.2f}")

    newpos = [xc_iter[-1], yc_iter[-1]]
    dx = x - position_xy[0]
    dy = y - position_xy[1]
    total = np.sqrt(dx**2 + dy**2)

    if verbose:
        print(f"Final shift: dx={dx:+.2f}, dy={dy:+.2f}, total={total:.2f}")

    if total > max_shift:
        warn(f"Object with initial position ({xc_iter[-1]}, {yc_iter[-1]}) "
             + f"shifted larger than {max_shift} ({total:.2f}).")

    # if verbose:
    #     print('Found centroid after {} iterations'.format(i_iter))
    #     print('Initially {}'.format(position_xy))
    #     print('Converged ({}, {})'.format(xc_iter[i_iter], yc_iter[i_iter]))
    #     shift = position_xy - np.array([xc_iter[i_iter], yc_iter[i_iter]])
    #     print('(Python/C-like indexing, not IRAF/FITS/Fortran)')
    #     print()
    #     print('Shifted to {}'.format(shift))
    #     print('\tShift tolerance was {}'.format(tol_shift))

    # if full:
    #     original_cut = Cutout2D(data=ccd.data,
    #                             position=position_xy,
    #                             size=cbox_size)
    #     final_cut = Cutout2D(data=ccd.data,
    #                          position=newpos,
    #                          size=cbox_size)
    #     return newpos, original_cut, final_cut

    if full:
        return newpos, np.array(xc_iter), np.array(yc_iter), total

    return newpos


def find_centroid_com(ccd, position_xy, maxiters=5, cbox_size=5., csigma=3.,
                      ssky=0, tol_shift=1.e-4, max_shift=1,
                      max_shift_step=None, verbose=False, full=False):
    ''' Find the intensity-weighted centroid iteratively.
    Simply run `centroiding_iteration` function iteratively for
    `maxiters` times. Given the initial guess of centroid position in
    image xy coordinate, it finds the intensity-weighted centroid
    (center of mass) after rejecting pixels by sigma-clipping.

    Parameters
    ----------
    ccd : CCDData or ndarray
        The whole image which the `position_xy` is calculated.

    position_xy : array-like
        The position of the initial guess in image XY coordinate.

    cbox_size : float or int, optional
        The size of the box to find the centroid. Recommended to use 2.5
        to 4.0 times the seeing FWHM.  Minimally about 5 pixel is
        recommended. If extended source (e.g., comet), recommend larger
        cbox.
        See:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?centerpars

    csigma : float or int, optional
        The parameter to use in sigma-clipping. Using pixels only above
        3-simga level for centroiding is recommended. See Ma+2009,
        Optics Express, 17, 8525.

    ssky : float, optional
        The sample standard deviation of the sky or background. The
        pixels above the local minima (of the array of size
        ``cbox_size``) plus ``csigma*ssky`` will be used for centroid,
        following the default of IRAF's ``noao.digiphot.apphot``.
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?centerpars.hlp

    tol_shift : float
        The absolute tolerance for the shift. If the shift in centroid
        after iteration is smaller than this, iteration stops.

    max_shift: float
        The maximum acceptable shift. If shift is larger than this,
        raises warning.

    max_shift_step : float, None, optional
        The maximum acceptable shift for each iteration. If the shift
        (say ``naive_shift``) is larger than this, the actual shift will
        be ``max_shift_step`` towards the direction identical to
        ``naive_shift``. If ``None`` (default), no such truncation is
        done.

    verbose : bool
        Whether to print how many iterations were needed for the
        centroiding.

    full : bool
        Whether to return the original and final cutout images.

    Returns
    -------
    com_xy : list
        The iteratively found centroid position.
    '''
    warn("DEPRECATED -- use find_centroid or find_center_2dg",
         AstropyDeprecationWarning)

    def _centroiding_iteration(ccd, position_xy, centroider=centroid_com,
                               cbox_size=5., csigma=3.,
                               max_shift_step=None, ssky=0, verbose=False):
        ''' Find the intensity-weighted centroid of the image iteratively

        Returns
        -------
        position_xy : float
            The centroided location in the original image coordinate
            in image XY.

        shift : float
            The total distance between the initial guess and the
            fitted centroid, i.e., the distance between `(xc_img,
            yc_img)` and `position_xy`.
        '''

        x_init, y_init = position_xy
        cutccd = Cutout2D(ccd.data, position=position_xy, size=cbox_size)
        cthresh = np.min(cutccd.data) + csigma * ssky
        # using pixels only above med + 3*std for centroiding is recommended.
        # See Ma+2009, Optics Express, 17, 8525
        # -- I doubt this... YPBach 2019-07-08 10:43:54 (KST: GMT+09:00)

        mask = (cutccd.data < cthresh)

        if verbose:
            n_all = np.size(mask)
            n_rej = np.count_nonzero(mask.astype(int))
            print(f"\t{n_rej} / {n_all} rejected [threshold = {cthresh:.3f} "
                  + f"from min ({np.min(cutccd.data):.3f}) "
                  + f"+ csigma ({csigma}) * ssky ({ssky:.3f})]")

        if ccd.mask is not None:
            mask += ccd.mask

        x_c_cut, y_c_cut = centroider(data=cutccd.data, mask=mask)
        # The position is in the cutout image coordinate, e.g., (3, 3).

        x_c, y_c = cutccd.to_original_position((x_c_cut, y_c_cut))
        # convert the cutout image coordinate to original coordinate.
        # e.g., (3, 3) becomes something like (137, 189)

        pos_new_naive = cutccd.to_original_position((x_c_cut, y_c_cut))
        # convert the cutout image coordinate to original coordinate.
        # e.g., (3, 3) becomes something like (137, 189)

        dx, dy, shift = scaling_shift(position_xy, pos_new_naive,
                                      max_shift_step=max_shift_step,
                                      verbose=verbose)
        pos_new = (position_xy[0] + dx, position_xy[1] + dy)

        return pos_new, shift

    if not isinstance(ccd, CCDData):
        _ccd = CCDData(ccd, unit='adu')  # Just a dummy
    else:
        _ccd = ccd.copy()

    i_iter = 0
    xc_iter = [position_xy[0]]
    yc_iter = [position_xy[1]]
    shift = []
    d = 0
    if verbose:
        print(f"Initial xy: ({xc_iter[0]}, {yc_iter[0]}) [0-index]")
        print(f"\t(max iteration {maxiters:d}, shift tolerance {tol_shift})")

    for i_iter in range(maxiters):
        xy_old = (xc_iter[-1], yc_iter[-1])
        if verbose:
            print(f"Iteration {i_iter+1:d} / {maxiters:d}: ")
        (x, y), d = _centroiding_iteration(ccd=_ccd,
                                           position_xy=xy_old,
                                           cbox_size=cbox_size,
                                           csigma=csigma,
                                           ssky=ssky,
                                           max_shift_step=max_shift_step,
                                           verbose=verbose)
        xc_iter.append(x)
        yc_iter.append(y)
        shift.append(d)
        if d < tol_shift:
            if verbose:
                print(f"Finishing iteration (shift {d:.5f} < tol_shift).")
            break
        if verbose:
            print(f"\t({x:.2f}, {y:.2f}), shifted {d:.2f}")

    newpos = [xc_iter[-1], yc_iter[-1]]
    dx = x - position_xy[0]
    dy = y - position_xy[1]
    total = np.sqrt(dx**2 + dy**2)

    if verbose:
        print(f"Final shift: dx={dx:+.2f}, dy={dy:+.2f}, total={total:.2f}")

    if total > max_shift:
        warn(f"Object with initial position ({xc_iter[-1]}, {yc_iter[-1]}) "
             + f"shifted larger than {max_shift} ({total:.2f}).")

    # if verbose:
    #     print('Found centroid after {} iterations'.format(i_iter))
    #     print('Initially {}'.format(position_xy))
    #     print('Converged ({}, {})'.format(xc_iter[i_iter], yc_iter[i_iter]))
    #     shift = position_xy - np.array([xc_iter[i_iter], yc_iter[i_iter]])
    #     print('(Python/C-like indexing, not IRAF/FITS/Fortran)')
    #     print()
    #     print('Shifted to {}'.format(shift))
    #     print('\tShift tolerance was {}'.format(tol_shift))

    # if full:
    #     original_cut = Cutout2D(data=ccd.data,
    #                             position=position_xy,
    #                             size=cbox_size)
    #     final_cut = Cutout2D(data=ccd.data,
    #                          position=newpos,
    #                          size=cbox_size)
    #     return newpos, original_cut, final_cut

    if full:
        return newpos, np.array(xc_iter), np.array(yc_iter), total

    return newpos
