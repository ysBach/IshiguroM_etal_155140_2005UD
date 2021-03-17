from warnings import warn
from astropy.visualization import (ImageNormalize, LinearStretch,
                                   ZScaleInterval, simple_norm)
__all__ = ["znorm", "zimshow", "norm_imshow"]


def znorm(image, stretch=LinearStretch(), **kwargs):
    return ImageNormalize(image,
                          interval=ZScaleInterval(**kwargs),
                          stretch=stretch)


def zimshow(ax, image, stretch=LinearStretch(), cmap=None, origin='lower',
            zscale_kw={}, **kwargs):
    im = ax.imshow(image,
                   norm=znorm(image, stretch=stretch, **zscale_kw),
                   origin=origin,
                   cmap=cmap,
                   **kwargs)
    return im


def norm_imshow(ax, data, origin='lower', stretch='linear', power=1.0,
                asinh_a=0.1, min_cut=None, max_cut=None, min_percent=None,
                max_percent=None, percent=None, clip=True, log_a=1000,
                zscale=False, vmin=None, vmax=None, **kwargs):
    """ Do normalization and do imshow
    """
    if zscale:
        zs = ImageNormalize(data, interval=ZScaleInterval())
        min_cut = zs.vmin
        max_cut = zs.vmax

    if vmin is not None and min_cut is not None:
        warn("vmin will override min_cut.")

    if vmax is not None and max_cut is not None:
        warn("vmax will override max_cut.")

    norm = simple_norm(data, stretch=stretch, power=power, asinh_a=asinh_a,
                       min_cut=vmin or min_cut, max_cut=vmax or max_cut,
                       min_percent=min_percent, max_percent=max_percent,
                       percent=percent, clip=clip, log_a=log_a)
    im = ax.imshow(data, norm=norm, origin=origin, **kwargs)
    return im
