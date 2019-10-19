import numpy as np
from astropy import units as u
from photutils import (CircularAnnulus, CircularAperture, EllipticalAnnulus,
                       EllipticalAperture, PixelAperture, RectangularAperture,
                       SkyAperture)
from photutils.aperture import ApertureMask
from photutils.aperture.attributes import (AngleOrPixelScalarQuantity,
                                           AngleScalarQuantity, PixelPositions,
                                           PositiveScalar, Scalar,
                                           SkyCoordPositions)

__all__ = ["cutout_from_ap", "ap_to_cutout_position",
           "circ_ap_an", "ellip_ap_an", "pill_ap_an",
           "PillBoxMaskMixin", "PillBoxAperture", "PillBoxAnnulus",
           "SkyPillBoxAperture", "SkyPillBoxAnnulus"]


def cutout_from_ap(ap, ccd):
    ''' Returns a Cutout2D object from bounding boxes of aperture/annulus.
    Parameters
    ----------
    ap : `photutils.Aperture`
        Aperture or annulus to cut the ccd.

    ccd : `astropy.nddata.CCDData` or ndarray
        The ccd to be cutout.
    '''
    from astropy.nddata import CCDData, Cutout2D
    if not isinstance(ccd, CCDData):
        ccd = CCDData(ccd, unit="adu")  # dummy unit

    positions = np.atleast_2d(ap.positions)
    try:
        bboxes = np.atleast_1d(ap.bbox)
    except AttributeError:
        bboxes = np.atleast_1d(ap.bounding_boxes)
    sizes = [bbox.shape for bbox in bboxes]
    cuts = []
    for pos, size in zip(positions, sizes):
        cuts.append(Cutout2D(ccd.data, position=pos, size=size))

    if len(cuts) == 1:
        return cuts[0]
    else:
        return cuts


def ap_to_cutout_position(ap, cutout2d):
    ''' Returns a new aperture/annulus only by updating ``positions``
    Parameters
    ----------
    ap : `photutils.Aperture`
        Aperture or annulus to update the ``.positions``.

    cutout2d : `astropy.nddata.Cutout2D`
        The cutout ccd to update ``ap.positions``.
    '''
    import copy
    newap = copy.deepcopy(ap)
    pos_old = np.atleast_2d(newap.positions)  # Nx2 positions
    newpos = []
    for pos in pos_old:
        newpos.append(cutout2d.to_cutout_position(pos))
    newap.positions = newpos
    return newap


"""
def cut_for_ap(to_move, based_on=None, ccd=None):
    ''' Cut ccd to ndarray from bounding box of ``based_on``.
    Useful for plotting aperture and annulus after cutting out centering
    on the object of interest.

    Parameters
    ----------
    to_move, based_on : `~photutils.Aperture`
        The aperture to be moved, and the reference.
    '''
    import copy

    moved = copy.deepcopy(to_move)

    if based_on is None:
        base = copy.deepcopy(to_move)
    else:
        base = copy.deepcopy(based_on)

    if np.atleast_2d(to_move.positions).shape[0] != 1:
        raise ValueError("multi-positions 'to_move' is not supported yet.")
    if np.atleast_2d(base.positions).shape[0] != 1:
        raise ValueError("multi-positions 'based_on' is not supported yet.")

    # for photutils before/after 0.7 compatibility...
    bbox = np.atleast_1d(base.bounding_boxes)[0]
    moved.positions = moved.positions - np.array([bbox.ixmin, bbox.iymin])

    if ccd is not None:
        from astropy.nddata import CCDData, Cutout2D
        if not isinstance(ccd, CCDData):
            ccd = CCDData(ccd, unit='adu')  # dummy unit
        # for photutils before/after 0.7 compatibility...
        pos = np.atleast_2d(moved.positions)[0]
        cut = Cutout2D(data=ccd.data, position=pos, size=bbox.shape)
        return moved, cut

    return moved

def cut_for_ap(to_move, based_on=None, ccd=None):
    ''' Cut ccd to ndarray from bounding box of ``based_on``.
    Useful for plotting aperture and annulus after cutting out centering
    on the object of interest.

    Parameters
    ----------
    to_move, based_on : `~photutils.Aperture`
        The aperture to be moved, and the reference.
    '''
    import copy

    moved = copy.deepcopy(to_move)

    if based_on is None:
        base = copy.deepcopy(to_move)
    else:
        base = copy.deepcopy(based_on)

    if ccd is not None:
        from astropy.nddata import CCDData
        if not isinstance(ccd, CCDData):
            ccd = CCDData(ccd, unit='adu')  # dummy unit

    pos_orig = np.atleast_2d(moved.positions)  # not yet moved
    pos_base = np.atleast_2d(base.positions)
    N_moved = pos_orig.shape[0]
    N_base = pos_base.shape[0]

    if N_base != 1 and N_moved != N_base:
        raise ValueError("based_on should have one 'positions' or "
                         + "have same number as 'move_to.positions'.")

    bboxes = np.atleast_1d(base.bounding_boxes)
    if base == 1:
        bboxes = np.repeat(bboxes, N_moved, 0)

    cuts = []
    for i, (position, bbox) in enumerate(zip(pos_orig, bboxes)):
        pos_cut = (position - np.array([bbox.ixmin, bbox.iymin]))

        moved.positions[i] = pos_cut
        if ccd is not None:
            from astropy.nddata import Cutout2D
            size = bbox.shape
            cut = Cutout2D(data=ccd.data, position=position, size=size)
            cuts.append(cut)

    if ccd is not None:
        if N_base == 1:
            return moved, cuts[0]
        else:
            return moved, cuts
    else:
        return moved
"""


def circ_ap_an(positions, fwhm, f_ap=1.5, f_in=4., f_out=6.):
    ''' A convenience function for pixel circular aperture/annulus
    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

        * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
        * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs
        * `~astropy.units.Quantity` instance of ``(x, y)`` pairs in
            pixel units

    fwhm : float
        The FWHM in pixel unit.

    f_ap, f_in, f_out: int or float, optional
        The factors multiplied to ``fwhm`` to set the aperture radius,
        inner sky radius, and outer sky radius, respectively. Defaults
        are ``1.5``, ``4.0``, and ``6.0``, respectively, which are de
        facto standard values used by classical IRAF users.

    Returns
    -------
    ap, an : `~photutils.CircularAperture` and `~photutils.CircularAnnulus`
        The object aperture and sky annulus.
    '''
    r_ap = f_ap * fwhm
    r_in = f_in * fwhm
    r_out = f_out * fwhm
    ap = CircularAperture(positions=positions, r=r_ap)
    an = CircularAnnulus(positions=positions, r_in=r_in, r_out=r_out)
    return ap, an


def ellip_ap_an(positions, fwhm, theta=0.,
                f_ap=(1.5, 1.5), f_in=(4., 4.), f_out=(6., 6.)):
    ''' A convenience function for pixel elliptical aperture/annulus
    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

        * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
        * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs
        * `~astropy.units.Quantity` instance of ``(x, y)`` pairs in
            pixel units

    fwhm : float
        The FWHM in pixel unit.

    theta : float, optional
        The rotation angle in radians of the ellipse semimajor axis from
        the positive ``x`` axis.  The rotation angle increases
        counterclockwise.  The default is 0.

    f_ap, f_in, f_out: int or float, list or tuple of such, optional
        The factors multiplied to ``fwhm`` to set the aperture ``a`` and
        ``b``, inner sky ``a`` and ``b``, and outer sky ``a`` and ``b``,
        respectively. If scalar, it is assumed to be identical for both
        ``a`` and ``b`` parameters. Defaults are ``(1.5, 1.5)``, ``(4.0,
        4.0)``, and ``(6.0, 6.0)``, respectively, which are de facto
        standard values used by classical IRAF users.

    Returns
    -------
    ap, an : `~photutils.EllipticalAperture` and `~photutils.EllipticalAnnulus`
        The object aperture and sky annulus.
    '''
    if np.isscalar(fwhm):
        fwhm = np.repeat(fwhm, 2)

    if np.isscalar(f_ap):
        f_ap = np.repeat(f_ap, 2)

    if np.isscalar(f_in):
        f_in = np.repeat(f_in, 2)

    if np.isscalar(f_out):
        f_out = np.repeat(f_out, 2)

    a_ap = f_ap[0] * fwhm[0]
    b_ap = f_ap[1] * fwhm[1]
    a_in = f_in[0] * fwhm[0]
    a_out = f_out[0] * fwhm[0]
    b_out = f_out[1] * fwhm[1]

    ap = EllipticalAperture(positions=positions, a=a_ap, b=b_ap, theta=theta)
    an = EllipticalAnnulus(positions=positions, a_in=a_in, a_out=a_out,
                           b_out=b_out, theta=theta)

    return ap, an


def pill_ap_an(positions, fwhm, trail, theta=0.,
               f_ap=(1.5, 1.5), f_in=(4., 4.), f_out=(6., 6.), f_w=1.):
    ''' A convenience function for pixel elliptical aperture/annulus
    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

        * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
        * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs
        * `~astropy.units.Quantity` instance of ``(x, y)`` pairs in
            pixel units

    fwhm : float
        The FWHM in pixel unit.

    trail : float
        The trail length in pixel unit. The trail is assumed to be
        extended along the ``x`` axis.

    theta : float, optional
        The rotation angle in radians of the ellipse semimajor axis from
        the positive ``x`` axis.  The rotation angle increases
        counterclockwise.  The default is 0.

    f_ap, f_in, f_out: int or float, list or tuple of such, optional
        The factors multiplied to ``fwhm`` to set the aperture ``a`` and
        ``b``, inner sky ``a`` and ``b``, and outer sky ``a`` and ``b``,
        respectively, for the elliptical component of the pill box. If
        scalar, it is assumed to be identical for both ``a`` and ``b``
        parameters. Defaults are ``(1.5, 1.5)``, ``(4.0, 4.0)``, and
        ``(6.0, 6.0)``, respectively, which are de facto standard values
        used by classical IRAF users.

    f_w : int or float
        The factor multiplied to ``trail`` to make a rectangular
        component of the pill box (both `~PillBoxAperture` and
        `~PillBoxAnnulus`). Note that this width is identical for both
        aperture and annulus.

    Returns
    -------
    ap, an : `~PillBoxAperture` and `~PillBoxAnnulus`
        The object aperture and sky annulus.
    '''
    if np.isscalar(fwhm):
        fwhm = np.repeat(fwhm, 2)

    if np.isscalar(f_ap):
        f_ap = np.repeat(f_ap, 2)

    if np.isscalar(f_in):
        f_in = np.repeat(f_in, 2)

    if np.isscalar(f_out):
        f_out = np.repeat(f_out, 2)

    a_ap = f_ap[0] * fwhm[0]
    b_ap = f_ap[1] * fwhm[1]
    a_in = f_in[0] * fwhm[0]
    a_out = f_out[0] * fwhm[0]
    b_out = f_out[1] * fwhm[1]

    w = f_w * trail

    ap = PillBoxAperture(positions=positions,
                         a=a_ap,
                         b=b_ap,
                         w=w,
                         theta=theta)
    an = PillBoxAnnulus(positions=positions,
                        a_in=a_in,
                        a_out=a_out,
                        b_out=b_out,
                        w=w,
                        theta=theta)
    return ap, an


"""
def set_pillbox_ap(positions, sigmas, ksigma=3, trail=0, theta=0):
    ''' Setup PillBoxAperture
    Parameters
    ----------
    positions : Nx2 array
        The positions in xy.

    sigmas : int, float, or length 2 of such
        The sigma or scale lengths along the major and minor axes of the
        trailed PSF. Order must be longer and then shorter.
    '''
    ksigma = np.atleast_1d(ksigma)
    if ksigma.shape[0] == 1:
        ksigma = ksigma.repeat(2)
    elif ksigma.shape[0] != 2 or ksigma.ndim != 1:
        raise TypeError("sigmas must be int or float of one or two elements. "
                        + f"Now it has shape = {ksigma.shape}")
    a = (ksigma[0]*sigmas[0] - trail) / 2 * (pix2arcsec)
    b = 3*sig_y_fit*pix2arcsec
    return PillBoxAperture(positions, trail, )
"""


class PillBoxMaskMixin:
    @property
    def _set_aperture_elements(self):
        """ Set internal aperture elements.
        ``self._ap_rect``, ``self.ap_el_1``, ``self.ap_el_2`` and their
        ``_in`` counterparts are always made by
        ``np.atleast_2d(self.position)``, so their results are always in
        the ``N x 2`` shape.
        """
        if hasattr(self, 'a'):
            w = self.w
            a = self.a
            b = self.b
            h = self.h
            theta = self.theta
        elif hasattr(self, 'a_in'):  # annulus
            w = self.w
            a = self.a_out
            b = self.b_out
            h = self.h_out
            theta = self.theta
        else:
            raise ValueError('Cannot determine the aperture shape.')

        # positions only accepted in the shape of (N, 2), so shape[0]
        # gives the number of positions:
        pos = np.atleast_2d(self.positions)
        self.offset = np.array([w*np.cos(theta)/2, w*np.sin(theta)/2])
        offsets = np.repeat(np.array([self.offset, ]), pos.shape[0], 0)

        # aperture elements for aperture,
        # OUTER aperture elements for annulus:
        self._ap_rect = RectangularAperture(positions=pos,
                                            w=w, h=h, theta=theta)
        self._ap_el_1 = EllipticalAperture(positions=pos - offsets,
                                           a=a, b=b, theta=theta)
        self._ap_el_2 = EllipticalAperture(positions=pos + offsets,
                                           a=a, b=b, theta=theta)

        if hasattr(self, 'a_in'):  # inner components of annulus
            self._ap_rect_in = RectangularAperture(positions=pos,
                                                   w=self.w,
                                                   h=self.h_in,
                                                   theta=self.theta)
            self._ap_el_1_in = EllipticalAperture(positions=pos - offsets,
                                                  a=self.a_in,
                                                  b=self.b_in,
                                                  theta=self.theta)
            self._ap_el_2_in = EllipticalAperture(positions=pos + offsets,
                                                  a=self.a_in,
                                                  b=self.b_in,
                                                  theta=self.theta)

    @staticmethod
    def _prepare_mask(bbox, ap_r, ap_1, ap_2, method, subpixels, min_mask=0):
        """ Make the pill box mask array.
        Note
        ----
        To make an ndarray to represent the overlapping mask, the three
        (a rectangular and two elliptical) apertures are generated, but
        parallely shifted such that the bounding box has ``ixmin`` and
        ``iymin`` both zero. Then proper mask is generated as an
        ndarray. It is then used by ``PillBoxMaskMixin.to_mask`` to make
        an ``ApertureMask`` object by combining this mask with the
        original bounding box.

        Parameters
        ----------
        bbox : `~photutils.BoundingBox`
            The bounding box of the original aperture.

        ap_r : `~photutils.RectangularAperture`
            The rectangular aperture of a pill box.

        ap_1, ap_2 : `~photutils.EllipticalAperture`
            The elliptical apertures of a pill box. The order of
            left/right ellipses is not important for this method.

        method : See `~photutils.PillBoxMaskMixin.to_mask`

        subpixels : See `~photutils.PillBoxMaskMixin.to_mask`

        min_mask : float, optional
            The mask values smaller than this value is ignored (set to
            0). This is required because the subtraction of elliptical
            and rectangular masks give some negative values. One can set
            it to be ``1/subpixels**2`` because ``RectangularAperture``
            does not support ``method='exact'`` yet.

        Returns
        -------
        mask_pill : ndarray
            The mask of the pill box.
        """
        aps = []
        for ap in [ap_r, ap_1, ap_2]:
            pos_cent = ap.positions
            tmp_cent = pos_cent - np.array([bbox.ixmin, bbox.iymin])
            if hasattr(ap, 'w'):
                tmp_ap = RectangularAperture(positions=tmp_cent,
                                             w=ap.w, h=ap.h, theta=ap.theta)
            else:
                tmp_ap = EllipticalAperture(positions=tmp_cent,
                                            a=ap.a, b=ap.b, theta=ap.theta)
            aps.append(tmp_ap)

        bbox_shape = bbox.shape

        mask_kw = dict(method=method, subpixels=subpixels)
        mask_r = (aps[0].to_mask(**mask_kw).to_image(bbox_shape))
        mask_1 = (aps[1].to_mask(**mask_kw).to_image(bbox_shape))
        mask_2 = (aps[2].to_mask(**mask_kw).to_image(bbox_shape))

        # Remove both machine epsilon artifact & negative mask values:
        mask_pill_1 = mask_1 - mask_r
        mask_pill_1[mask_pill_1 < min_mask] = 0
        mask_pill_2 = mask_2 - mask_r
        mask_pill_2[mask_pill_2 < min_mask] = 0

        mask_pill = mask_r + mask_pill_1 + mask_pill_2

        # Overlap of elliptical parts may make value > 1:
        mask_pill[mask_pill > 1] = 1

        return mask_pill

    def to_mask(self, method='exact', subpixels=5):
        """ Return a mask for the aperture.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on
            the pixel grid.  Not all options are available for all
            aperture types.  Note that the more precise methods are
            generally slower.  The following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and
                  each pixel is calculated.  The returned mask will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture.  The returned mask will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``:
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending on
                  whether its center is in or out of the aperture.  If
                  ``subpixels=1``, this method is equivalent to
                  ``'center'``.  The returned mask will contain values
                  between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor
            in each dimension.  That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        Returns
        -------
        mask : `~photutils.ApertureMask` or list of `~photutils.ApertureMask`
            A mask for the aperture.  If the aperture is scalar then a
            single `~photutils.ApertureMask` is returned, otherwise a
            list of `~photutils.ApertureMask` is returned.
        """
        _, subpixels = self._translate_mask_mode(method, subpixels)
        min_mask = min(1.e-6, 1/(subpixels**2))
        masks = []
        try:
            bboxes = np.atleast_1d(self.bbox)
        except AttributeError:
            bboxes = np.atleast_1d(self.bounding_boxes)
        is_annulus = True if hasattr(self, 'a_in') else False

        for i, (bbox, ap_r, ap_1, ap_2) in enumerate(zip(bboxes,
                                                         self._ap_rect,
                                                         self._ap_el_1,
                                                         self._ap_el_2)):
            mask = self._prepare_mask(bbox, ap_r=ap_r, ap_1=ap_1, ap_2=ap_2,
                                      method=method,
                                      subpixels=subpixels,
                                      min_mask=min_mask)

            if is_annulus:
                mask -= self._prepare_mask(bbox,
                                           ap_r=self._ap_rect_in[i],
                                           ap_1=self._ap_el_1_in[i],
                                           ap_2=self._ap_el_2_in[i],
                                           method=method,
                                           subpixels=subpixels,
                                           min_mask=min_mask)

            masks.append(ApertureMask(mask, bbox))

        if self.isscalar:
            return masks[0]
        else:
            return masks

    @staticmethod
    def _pill_patches(ellipse_1, ellipse_2, **patch_kwargs):
        """ Make matplotlib.patches from ellipses.
        """
        import matplotlib.patches as mpatches
        import matplotlib.path as mpath

        path_1 = ellipse_1.get_path()
        tran_1 = ellipse_1.get_transform()
        trpath_1 = tran_1.transform_path(path_1)
        trpath_1_v = trpath_1.vertices
        trpath_1_c = trpath_1.codes
        pill_1_v = trpath_1_v[:len(trpath_1_v)//2, :]
        pill_1_c = trpath_1_c[:len(trpath_1_c)//2]

        path_2 = ellipse_2.get_path()
        tran_2 = ellipse_2.get_transform()
        trpath_2 = tran_2.transform_path(path_2)
        trpath_2_v = trpath_2.vertices
        trpath_2_c = trpath_2.codes
        pill_2_v = trpath_2_v[-(len(trpath_2_v)//2 + 1):, :]
        pill_2_c = trpath_2_c[-(len(trpath_2_c)//2 + 1):]

        pill_v = np.concatenate([pill_1_v, pill_2_v])
        pill_c = np.concatenate([pill_1_c, [mpath.Path.LINETO], pill_2_c[1:]])
        pill_path = mpath.Path(pill_v, pill_c)
        pill_patch = mpatches.PathPatch(pill_path, **patch_kwargs)

        return pill_patch


class PillBoxAperture(PillBoxMaskMixin, PixelAperture):
    """ A pill box aperture defined in pixel coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    """
    _shape_params = ('w', 'a', 'b', 'theta')
    positions = PixelPositions('positions')
    w = PositiveScalar('w')
    a = PositiveScalar('a')
    b = PositiveScalar('b')
    theta = Scalar('theta')

    def __init__(self, positions, w, a, b, theta=0.):
        self.positions = positions
        self.w = w
        self.a = a
        self.b = b
        self.h = 2*b
        self.theta = theta
        self._set_aperture_elements

    @property
    def _xy_extents(self):
        return np.abs(self.offset) + self._ap_el_1._xy_extents

    # def bounding_boxes(self):
    #     try:
    #         bboxes_rect = self._ap_rect.bbox
    #         bboxes_el_1 = self._ap_el_1.bbox
    #         bboxes_el_2 = self._ap_el_2.bbox
    #     except AttributeError:
    #         bboxes_rect = self._ap_rect.bounding_boxes
    #         bboxes_el_1 = self._ap_el_1.bounding_boxes
    #         bboxes_el_2 = self._ap_el_2.bounding_boxes

    #     bboxes = []
    #     for bb_r, bb_1, bb_2 in zip(bboxes_rect, bboxes_el_1, bboxes_el_2):
    #         bbox = (bb_r) | (bb_1) | (bb_2)
    #         bboxes.append(  )

    #     if self.isscalar:
    #         return bboxes[0]
    #     else:
    #         return bboxes

    @property
    def area(self):
        return self.w * self.h + np.pi * self.a * self.b

    def _to_patch(self, origin=(0, 0), indices=None, **kwargs):
        """
        """
        import matplotlib.patches as mpatches

        # xy_positions is already atleast_2d'ed.
        xy_positions, patch_kwargs = self._define_patch_params(
            origin=origin, indices=indices, **kwargs)

        patches = []
        theta_deg = self.theta * 180. / np.pi

        for xy_position in xy_positions:
            # The ellipse on the "right" whan theta = 0
            ellipse_1 = mpatches.Ellipse(xy_position + self.offset,
                                         2.*self.a,
                                         2.*self.b,
                                         theta_deg)
            # The ellipse on the "left" whan theta = 0
            ellipse_2 = mpatches.Ellipse(xy_position - self.offset,
                                         2.*self.a,
                                         2.*self.b,
                                         theta_deg)
            p = self._pill_patches(ellipse_1, ellipse_2, **patch_kwargs)

            patches.append(p)

        if self.isscalar:
            return patches[0]
        else:
            return patches

    def to_sky(self, wcs, mode='all'):
        """
        Convert the aperture to a `SkyPillBoxAperture` object
        defined in celestial coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or only including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `SkyPillBoxAperture` object
            A `SkyPillBoxAperture` object.
        """

        sky_params = self._to_sky_params(wcs, mode=mode)
        return SkyPillBoxAperture(**sky_params)


class PillBoxAnnulus(PillBoxMaskMixin, PixelAperture):
    """
    """
    _shape_params = ('w', 'a_in', 'a_out', 'b_out', 'theta')
    positions = PixelPositions('positions')
    w = PositiveScalar('w')
    a_in = PositiveScalar('a_in')
    a_out = PositiveScalar('a_out')
    b_out = PositiveScalar('b_out')
    theta = Scalar('theta')

    def __init__(self, positions, w, a_in, a_out, b_out, theta=0.):
        self.positions = positions
        self.w = w
        self.a_out = a_out
        self.a_in = a_in
        self.b_out = b_out
        self.b_in = self.b_out * self.a_in / self.a_out
        self.h_out = self.b_out*2
        self.h_in = self.b_in*2
        self.theta = theta
        self._set_aperture_elements

    @property
    def _xy_extents(self):
        return np.abs(self.offset) + self._ap_el_1._xy_extents

    # def bounding_boxes(self):
    #     """
    #     A list of minimal bounding boxes (`~photutils.BoundingBox`), one
    #     for each position, enclosing the exact elliptical apertures.
    #     """
    #     bboxes_rect = self._ap_rect.bounding_boxes
    #     bboxes_el_1 = self._ap_el_1.bounding_boxes
    #     bboxes_el_2 = self._ap_el_2.bounding_boxes
    #     bboxes = []
    #     for bb_r, bb_1, bb_2 in zip(bboxes_rect, bboxes_el_1, bboxes_el_2):
    #         bboxes.append( (bb_r) | (bb_1) | (bb_2) )

    #     if self.isscalar:
    #         return bboxes[0]
    #     else:
    #         return bboxes

    @property
    def area(self):
        return (self.w * (self.h_out - self.h_in)
                + np.pi * (self.a_out * self.b_out - self.a_in * self.b_in))

    def _to_patch(self, origin=(0, 0), indices=None, **kwargs):
        import matplotlib.patches as mpatches
        # xy_positions is already atleast_2d'ed.
        xy_positions, patch_kwargs = self._define_patch_params(
            origin=origin, indices=indices, **kwargs)

        patches = []
        theta_deg = self.theta * 180. / np.pi

        for xy_position in xy_positions:
            # The ellipse on the "right" whan theta = 0
            ellipse_1_in = mpatches.Ellipse(xy_position + self.offset,
                                            2.*self.a_in,
                                            2.*self.b_in,
                                            theta_deg)
            # The ellipse on the "left" whan theta = 0
            ellipse_2_in = mpatches.Ellipse(xy_position - self.offset,
                                            2.*self.a_in,
                                            2.*self.b_in,
                                            theta_deg)
            p_inner = self._pill_patches(ellipse_1_in, ellipse_2_in)

            # The ellipse on the "right" whan theta = 0
            ellipse_1_out = mpatches.Ellipse(xy_position + self.offset,
                                             2.*self.a_out,
                                             2.*self.b_out,
                                             theta_deg)
            # The ellipse on the "left" whan theta = 0
            ellipse_2_out = mpatches.Ellipse(xy_position - self.offset,
                                             2.*self.a_out,
                                             2.*self.b_out,
                                             theta_deg)
            p_outer = self._pill_patches(ellipse_1_out, ellipse_2_out)

            p = self._make_annulus_path(p_inner, p_outer)
            patches.append(mpatches.PathPatch(p, **patch_kwargs))

        if self.isscalar:
            return patches[0]
        else:
            return patches

    def to_sky(self, wcs, mode='all'):
        """
        Convert the aperture to a `SkyPillBoxAnnulus` object defined
        in celestial coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or only including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `SkyPillBoxAnnulus` object
            A `SkyPillBoxAnnulus` object.
        """
        sky_params = self._to_sky_params(wcs, mode=mode)
        return SkyPillBoxAnnulus(**sky_params)


class SkyPillBoxAperture(SkyAperture):
    """ A pill box aperture defined in sky coordinates.
    """
    _shape_params = ('w', 'a', 'b', 'theta')
    positions = SkyCoordPositions('positions')
    w = AngleOrPixelScalarQuantity('w')
    a = AngleOrPixelScalarQuantity('a')
    b = AngleOrPixelScalarQuantity('b')
    theta = AngleScalarQuantity('theta')

    def __init__(self, positions, w, a, b, theta=0.*u.deg):
        if not (w.unit.physical_type == a.unit.physical_type
                == b.unit.physical_type):
            raise ValueError("'w', 'a', and 'b' should all be angles or "
                             "in pixels")

        self.positions = positions
        self.w = w
        self.a = a
        self.b = b
        self.theta = theta

    def to_pixel(self, wcs, mode='all'):
        """
        Convert the aperture to an `PillBoxAperture` object defined
        in pixel coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or only including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `PillBoxAperture` object
            An `PillBoxAperture` object.
        """
        pixel_params = self._to_pixel_params(wcs, mode=mode)
        return PillBoxAperture(**pixel_params)


class SkyPillBoxAnnulus(SkyAperture):
    _shape_params = ('w', 'a_in', 'a_out', 'b_out', 'theta')
    positions = SkyCoordPositions('positions')
    w = AngleOrPixelScalarQuantity('w')
    a_in = AngleOrPixelScalarQuantity('a_in')
    a_out = AngleOrPixelScalarQuantity('a_out')
    b_out = AngleOrPixelScalarQuantity('b_out')
    theta = AngleScalarQuantity('theta')

    def __init__(self, positions, w, a_in, a_out, b_out, theta=0.*u.deg):
        if not (w.unit.physical_type == a_in.unit.physical_type
                == a_out.unit.physical_type == b_out.unit.physical_type):
            raise ValueError("'w', 'a_in', 'a_out', and 'b_out' should all be "
                             "angles or in pixels")

        self.positions = positions
        self.w = w
        self.a_out = a_out
        self.a_in = a_in
        self.b_out = b_out
        self.b_in = self.b_out * self.a_in / self.a_out
        self.h_out = self.b_out*2
        self.h_in = self.b_in*2
        self.theta = theta

    def to_pixel(self, wcs, mode='all'):
        """
        Convert the aperture to an `PillBoxAnnulus` object defined in
        pixel coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        mode : {'all', 'wcs'}, optional
            Whether to do the transformation including distortions
            (``'all'``; default) or only including only the core WCS
            transformation (``'wcs'``).

        Returns
        -------
        aperture : `PillBoxAnnulus` object
            An `PillBoxAnnulus` object.
        """
        pixel_params = self._to_pixel_params(wcs, mode=mode)
        return PillBoxAnnulus(**pixel_params)
