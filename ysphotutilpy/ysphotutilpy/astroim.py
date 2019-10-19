from pathlib import Path
from astropy import units as u
from astropy.wcs import WCS
import ysfitsutilpy as yfu

__all__ = ["AstroImage"]


class AstroImage:
    def __init__(self, fpath, target=None, gain_key="GAIN",
                 rdnoise_key="RDNOISE"):
        self.fpath = Path(fpath)
        self.ccd = yfu.load_ccd(fpath)
        self.header = self.ccd.header
        self.wcs = WCS(self.header)

        self.target = target
        if self.target is None:
            try:
                self.target = self.header["OBJECT"]
            except KeyError:
                pass

        self.gain = yfu.get_from_header(self.header, key=gain_key,
                                        unit=u.electron/u.adu,
                                        verbose=False, default=1)
        self.rdnoise = yfu.get_from_header(self.header, key=rdnoise_key,
                                           unit=u.electron,
                                           verbose=False, default=0)

    def calc_error(self, flat_err=0.):
        self.error = yfu.make_errmap(ccd=self.ccd,
                                     gain_epadu=self.gain.value,
                                     rdnoise_electron=self.rdnoise.value,
                                     flat_err=flat_err)


class TrailedImage(AstroImage):
    pass
