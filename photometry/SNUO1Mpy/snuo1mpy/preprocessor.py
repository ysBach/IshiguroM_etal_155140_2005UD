import pickle
from pathlib import Path
from warnings import warn

import pandas as pd
from astropy.io import fits
from astropy.io.fits import Card
from astropy.nddata import CCDData
from astropy.time import Time

from .utils import KEYMAP, MEDCOMB_KEYS, USEFUL_KEYS, cards_gain_rdnoise

try:
    import ysfitsutilpy as yfu
except ImportError:
    raise ImportError(
        "Please install ysfitsutilspy at: https://github.com/ysBach/ysfitsutilpy")


__all__ = ["Preprocessor"]


class Preprocessor():
    def __init__(self, topdir, rawdir, instrument="STX16803",
                 bias_type_key=["OBJECT"], bias_type_val=["bias"],
                 bias_group_key=[],
                 dark_type_key=["OBJECT"], dark_type_val=["dark"],
                 dark_group_key=["EXPTIME"],
                 flat_type_key=["OBJECT"], flat_type_val=["skyflat"],
                 flat_group_key=["FILTER"],
                 summary_keywords=USEFUL_KEYS):
        """
        Parameters
        ----------
        topdir : path-like
            The top directory of which all the other paths will be
            represented relative to.

        rawdir : path-like
            The directory where all the FITS files are stored (without
            any subdirectory)

        instrument : str
            The name of the instrument.

        xxxx_type_key : str or list of str, optional
            The header keys to be used for the identification of the
            xxxx frames (xxxx is one of bias, dark, flat). Each value
            should correspond to the same-index element of ``type_val``.
            Usually we use ``"OBJECT"`` or ``"IMAGETYP"``.

        xxxx_type_val : str, float, int or list of such, optional
            The header key and values to identify the xxxx frames (xxxx
            is one of bias, dark, flat) frames. Each value should
            correspond to the same-index element of ``type_key``.

        xxxx_group_key : str or list str, optional
            The header keywords to be used for grouping xxxx frames
            (xxxx is one of bias, dark, flat). For dark frames, usual
            choice can be ``['EXPTIME']``, and for flat frames,
            ``["FILTER"]``.

        summary_keywords : list of str, optional
            The keywords of the header to be used for the summary table.
        """
        topdir = Path(topdir)
        self.topdir = topdir  # e.g., Path('180412')
        self.rawdir = rawdir  # e.g., Path('180412/rawdata')
        self.listdir = self.topdir / "lists"
        self.instrument = instrument
        self.rawpaths = list(Path(rawdir).glob('*.fit'))
        self.rawpaths.sort()
        self.summary_keywords = summary_keywords
        self.newpaths = None
        self.summary_raw = None
        self.summary_red = None
        self.objpaths = None
        self.reducedpaths = None
        self.biaspaths = None
        self.darkpaths = None
        self.flatpaths = None
        # rawpaths: Original file paths
        # newpaths: Renamed paths
        # bias/dark/flatpaths: the dict that contains the paths to B/D/F.
        #   The keys of dict will be B/D/F_group_key and values will be the
        #   corresponding header values. These keys are NOT _group_key !!!
        #   (see below for the differences of _key and _group_key)

        if not set(bias_group_key).issubset(set(dark_group_key)):
            raise KeyError(
                "bias_grouped_key must be a subset of dark_group_key.")

        if not set(bias_group_key).issubset(set(flat_group_key)):
            raise KeyError(
                "bias_grouped_key must be a subset of flat_group_key.")

        _exptime_keys = ["EXPTIME", "EXPOSURE", "EXPOS"]
        if len(set(_exptime_keys).intersection(set(dark_group_key))) == 0:
            warn("dark_group_key does not seem to contain exposure time!")

        # xyz = <bias/dark/flat> <type/group> <key/value>, e.g., btk.
        btk, btv, bgk = yfu.chk_keyval(type_key=bias_type_key,
                                       type_val=bias_type_val,
                                       group_key=bias_group_key)
        dtk, dtv, dgk = yfu.chk_keyval(type_key=dark_type_key,
                                       type_val=dark_type_val,
                                       group_key=dark_group_key)
        ftk, ftv, fgk = yfu.chk_keyval(type_key=flat_type_key,
                                       type_val=flat_type_val,
                                       group_key=flat_group_key)
        self.bias_type_key = btk
        self.bias_type_val = btv
        self.bias_group_key = bgk
        self.bias_key = btk + bgk
        self.dark_type_key = dtk
        self.dark_type_val = dtv
        self.dark_group_key = dgk
        self.dark_key = dtk + dgk
        self.flat_type_key = ftk
        self.flat_type_val = ftv
        self.flat_group_key = fgk
        self.flat_key = ftk + fgk
        # NOTE: _key contains both the keys of _type_key and _group_key.
        #   I used this convention throughout the code.

    def initialize_self(self):
        ''' Initialization may convenient when process was halted amid.
        '''
        if self.summary_red is None:
            try:
                self.summary_red = pd.read_csv(
                    str(self.topdir / "summary_reduced.csv"))
                self.reducedpaths = self.summary_red["file"].tolist()
            except FileNotFoundError:
                pass

        if self.summary_raw is None:
            try:
                self.summary_raw = pd.read_csv(
                    str(self.topdir / "summary_raw.csv"))
                self.newpaths = self.summary_raw["file"].tolist()
            except FileNotFoundError:
                pass

        if self.newpaths is None:
            try:
                with open(self.listdir / "newpaths.pkl", 'rb') as pkl:
                    self.newpaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

        if self.objpaths is None:
            try:
                with open(self.listdir / "objpaths.pkl", 'rb') as pkl:
                    self.objpaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

        if self.biaspaths is None:
            try:
                with open(self.listdir / "biaspaths.pkl", 'rb') as pkl:
                    self.biaspaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

        if self.darkpaths is None:
            try:
                with open(self.listdir / "darkpaths.pkl", 'rb') as pkl:
                    self.darkpaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

        if self.flatpaths is None:
            try:
                with open(self.listdir / "flatpaths.pkl", 'rb') as pkl:
                    self.flatpaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

    def organize_raw(self,
                     rename_by=["OBSCAM", "OBJECT", "XBINNING", "YBINNING",
                                "YMD-HMS", "FILTER", "EXPTIME"],
                     mkdir_by=["OBJECT"], delimiter='-',
                     archive_dir=None, verbose=False):
        ''' Rename FITS files after updating theur headers.
        Parameters
        ----------
        rename_by : list of str
            The keywords in header to be used for the renaming of FITS
            files. Each keyword values are connected by ``delimiter``.

        mkdir_by : list of str, optional
            The keys which will be used to make subdirectories to
            classify files. If given, subdirectories will be made with
            the header value of the keys.

        delimiter : str, optional
            The delimiter for the renaming.

        archive_dir : path-like or None, optional
            Where to move the original FITS file. If ``None``, the
            original file will remain there. Deleting original FITS is
            dangerous so it is only supported to move the files. You may
            delete files manually if needed.
        '''

        newpaths = []
        objpaths = []
        uselessdir = self.rawdir / "useless"
        yfu.mkdir(uselessdir)
        yfu.mkdir(self.listdir)

        str_imgtyp = ("{:s}: IMAGETYP in header ({:s}) and that inferred from"
                      + "the filename ({:s}) doesn't seem to match.")
        str_useless = "{} is not a regular name. Moving to {}. "
        str_obj = ("{:s}: OBJECT in header({:s}) != filename({:s}). "
                   + "OBJECT in header is updated to match the filename.")
        # NOTE: it is better to give the filename a higher priority because
        #   it is easier to change filename than FITS header.

        for fpath in self.rawpaths:
            if fpath.name.startswith("CCD Image"):
                # The image not taken correctly are saved as dummy name
                # "CCD Image xxx.fit". It is user's fault to have this
                # kind of image, so move it to useless.
                print(str_useless.format(fpath.name, uselessdir))
                fpath.rename(uselessdir / fpath.name)
                continue

            # else:
            try:
                # Use `rsplit` because sometimes there are objnames like
                # `sa101-100`, i.e., includes the hyphen.
                # filt_or_bd : B/V/R/I/Ha/Sii/Oiii or bias/dkXX (XX=EXPTIME)
                hdr = fits.getheader(fpath)
                sp = fpath.name.rsplit('-')
                if len(sp) == 1:
                    sp = fpath.name.rsplit('_')
                obj_raw = sp[0]
                counter = sp[-1].split('.')[0][:4]
                filt_bd = sp[-1].split('.')[0][4:]
                filt_bd_low = filt_bd.lower()
                if obj_raw.lower() == 'cali':
                    if filt_bd_low.startswith("b"):
                        imgtyp = "bias"
                    elif filt_bd_low.startswith("d"):
                        imgtyp = "dark"
                    else:
                        print(str_useless.format(fpath.name, uselessdir))
                        fpath.rename(uselessdir / fpath.name)
                else:
                    imgtyp = hdr["IMAGETYP"]
                    
            except IndexError:
                print(str_useless.format(fpath.name, uselessdir))
                fpath.rename(uselessdir / fpath.name)
                continue

            cards_to_add = []

            # Update header OBJECT cuz it is super messy...
            #   Bias / Dark: understood from header IMAGETYP
            #   Dome / Sky flat / Object frame : understood from filename
            if imgtyp.lower() in ["bias", "bias frame"]:
                if not filt_bd_low.startswith("b"):
                    warn(str_imgtyp.format(fpath.name, imgtyp, filt_bd_low))
                obj = "bias"

            elif imgtyp.lower() in ["dark", "dark frame"]:
                if not filt_bd_low.startswith("d"):
                    warn(str_imgtyp.format(fpath.name, imgtyp, filt_bd_low))
                obj = "dark"

            elif obj_raw.lower() in ["skyflat", "domeflat"]:
                obj = obj_raw.lower()

            elif imgtyp.lower() in ["flat", "flat field"]:
                obj = "flat"

            else:
                if obj_raw != str(hdr[KEYMAP["OBJECT"]]):
                    warn(str_obj.format(fpath.name,
                                        hdr[KEYMAP["OBJECT"]], obj_raw))
                obj = obj_raw

            hdr[KEYMAP["OBJECT"]] = obj

            # Add gain and rdnoise:
            grdcards = cards_gain_rdnoise(instrument=self.instrument)
            [cards_to_add.append(c) for c in grdcards]

            # Add counter if there is none:
            if "COUNTER" not in hdr:
                cards_to_add.append(Card("COUNTER", counter, "Image counter"))

            # Add unit if there is none:
            if "BUNIT" not in hdr:
                cards_to_add.append(Card("BUNIT", "ADU", "Pixel value unit"))

            # Calculate airmass except for bias/dark
            if obj not in ["bias", "dark"]:
                # FYI: flat require airmass just for check (twilight/night)
                try:
                    hdr = yfu.airmass_from_hdr(hdr,
                                               ra_key="OBJCTRA",
                                               dec_key="OBJCTDEC",
                                               ut_key=KEYMAP["DATE-OBS"],
                                               exptime_key=KEYMAP["EXPTIME"],
                                               lon_key="SITELONG",
                                               lat_key="SITELAT",
                                               height_key="HEIGHT",
                                               equinox="J2000",
                                               frame='icrs',
                                               height=147,
                                               return_header=True)

                except KeyError:
                    if verbose:
                        print(f"{fpath} failed in airmass calculation: "
                              + "KeyError")

            datetime = Time(hdr[KEYMAP["DATE-OBS"]]).strftime("%Y%m%d-%H%M%S")
            obscam = f"SNUO_{self.instrument}"

            # Add YMD-HMS, and OBS-CAM
            cards_to_add.append(Card("YMD-HMS", datetime, "YYYYmmdd-HHMMSS"))
            cards_to_add.append(
                Card("OBSCAM", obscam, "<observatory>_<camera>"))

            add_hdr = fits.Header(cards_to_add)

            newpath = yfu.fitsrenamer(fpath,
                                      header=hdr,
                                      rename_by=rename_by,
                                      delimiter=delimiter,
                                      add_header=add_hdr,
                                      mkdir_by=mkdir_by,
                                      archive_dir=archive_dir,
                                      key_deprecation=True,
                                      keymap=KEYMAP,
                                      verbose=verbose)

            newpaths.append(newpath)
            if obj not in ["flat", "skyflat", "domeflat", "bias", "dark"]:
                objpaths.append(newpath)

        # Save list of file paths for future use.
        # It doesn't take much storage and easy to erase if you want.
        with open(self.listdir / 'newpaths.list', 'w+') as ll:
            for p in newpaths:
                ll.write(f"{str(p)}\n")

        with open(self.listdir / 'objpaths.list', 'w+') as ll:
            for p in objpaths:
                ll.write(f"{str(p)}\n")

        # Python specific pickle
        with open(self.listdir / 'newpaths.pkl', 'wb') as pkl:
            pickle.dump(newpaths, pkl)

        with open(self.listdir / 'objpaths.pkl', 'wb') as pkl:
            pickle.dump(objpaths, pkl)

        self.newpaths = newpaths
        self.objpaths = objpaths
        self.summary_raw = yfu.make_summary(
            newpaths,
            output=self.topdir/"summary_raw.csv",
            keywords=self.summary_keywords,
            pandas=True,
            verbose=verbose
        )

    def make_bias(self, savedir=None, delimiter='-', dtype='float32',
                  comb_kwargs=MEDCOMB_KEYS):
        ''' Finds and make bias frames.
        Parameters
        ----------
        savedir : path-like, optional.
            The directory where the frames will be saved.

        delimiter : str, optional.
            The delimiter for the renaming.

        dtype : str or numpy.dtype object, optional.
            The data type you want for the final master bias frame. It is
            recommended to use ``float32`` or ``int16`` if there is no
            specific reason.

        comb_kwargs: dict or None, optional.
            The parameters for `~ysfitsutilpy.combine_ccd`.
        '''
        # Initial settings
        self.initialize_self()

        if savedir is None:
            savedir = self.topdir

        yfu.mkdir(Path(savedir))
        biaspaths = {}
        # For simplicity, crop the original data by type_key and
        # type_val first.
        st = self.summary_raw.copy()
        for k, v in zip(self.bias_type_key, self.bias_type_val):
            st = st[st[k] == v]

        # For grouping, use _key (i.e., type_key + group_key). This is
        # because (1) it is not harmful cuz type_key will have unique
        # column values as ``st`` has already been cropped in above for
        # loop (2) by doing this we get more information from combining
        # process because, e.g., "images with ["OBJECT", "EXPTIME"] =
        # ["dark", 1.0] are loaded" will be printed rather than just
        # "images with ["EXPTIME"] = [1.0] are loaded".
        gs = st.groupby(self.bias_key)

        # Do bias combine:
        for bias_val, bias_group in gs:
            if not isinstance(bias_val, tuple):
                bias_val = tuple([str(bias_val)])
            fname = delimiter.join([str(x) for x in bias_val]) + ".fits"
            fpath = Path(savedir) / fname
            _ = yfu.combine_ccd(bias_group["file"].tolist(),
                                output=fpath,
                                dtype=dtype,
                                **comb_kwargs,
                                type_key=self.bias_key,
                                type_val=bias_val)
            biaspaths[tuple(bias_val)] = fpath

        # Save list of file paths for future use.
        # It doesn't take much storage and easy to erase if you want.
        with open(self.listdir / 'biaspaths.list', 'w+') as ll:
            for p in list(biaspaths.values()):
                ll.write(f"{str(p)}\n")

        with open(self.listdir / 'biaspaths.pkl', 'wb') as pkl:
            pickle.dump(biaspaths, pkl)

        self.biaspaths = biaspaths

    def make_dark(self, savedir=None, do_bias=True, mbiaspath=None,
                  dtype='float32', delimiter='-', comb_kwargs=MEDCOMB_KEYS):
        """ Makes and saves dark (bias subtracted) images.
        Parameters
        ----------
        savedir: path-like, optional
            The directory where the frames will be saved.

        do_bias : bool, optional
            If ``True``, subtracts bias from dark frames using
            self.biaspahts. You can also specify ``mbiaspath`` to ignore
            that in ``self.``.

        mbiaspath : None, path-like, optional
            If you want to force a certain bias to be used, then you can
            specify its path here.

        delimiter : str, optional.
            The delimiter for the renaming.

        dtype : str or numpy.dtype object, optional.
            The data type you want for the final master bias frame. It
            is recommended to use ``float32`` or ``int16`` if there is
            no specific reason.

        comb_kwargs : dict or None, optional
            The parameters for ``combine_ccd``.
        """
        # Initial settings
        self.initialize_self()

        if savedir is None:
            savedir = self.topdir

        yfu.mkdir(Path(savedir))
        darkpaths = {}

        # For simplicity, crop the original data by type_key and
        # type_val first.
        st = self.summary_raw.copy()
        for k, v in zip(self.dark_type_key, self.dark_type_val):
            st = st[st[k] == v]

        # For grouping, use _key (i.e., type_key + group_key). This is
        # because (1) it is not harmful cuz type_key will have unique
        # column values as ``st`` has already been cropped in above for
        # loop (2) by doing this we get more information from combining
        # process because, e.g., "images with ["OBJECT", "EXPTIME"] =
        # ["dark", 1.0] are loaded" will be printed rather than just
        # "images with ["EXPTIME"] = [1.0] are loaded".
        gs = st.groupby(self.dark_key)

        # Do dark combine:
        for dark_val, dark_group in gs:
            if not isinstance(dark_val, tuple):
                dark_val = tuple([dark_val])
            fname = delimiter.join([str(x) for x in dark_val]) + ".fits"
            fpath = Path(savedir) / fname

            mdark = yfu.combine_ccd(dark_group["file"].tolist(),
                                    output=None,
                                    dtype=dtype,
                                    **comb_kwargs,
                                    type_key=self.dark_key,
                                    type_val=dark_val)

            # set path to master bias
            if mbiaspath is not None:
                biaspath = mbiaspath
            elif do_bias:
                # corresponding key for biaspaths:
                corr_bias = tuple(self.bias_type_val)
                # if _group_key not empty, add appropriate ``group_val``:
                if self.bias_group_key:
                    corr_bias += tuple(dark_group[self.bias_group_key].iloc[0])
                # else: empty. path is fully specified by _type_val.
                try:
                    biaspath = self.biaspaths[corr_bias]
                except KeyError:
                    biaspath = None
                    warn(f"Bias not available for {corr_bias}. "
                         + "Processing without bias.")

            mdark = yfu.bdf_process(mdark,
                                    mbiaspath=biaspath,
                                    dtype=dtype,
                                    unit=None)

            mdark.write(fpath, output_verify='fix', overwrite=True)
            darkpaths[tuple(dark_val)] = fpath

        # Save list of file paths for future use.
        # It doesn't take much storage and easy to erase if you want.
        with open(self.listdir / 'darkpaths.list', 'w+') as ll:
            for p in list(darkpaths.values()):
                ll.write(f"{str(p)}\n")

        with open(self.listdir / 'darkpaths.pkl', 'wb') as pkl:
            pickle.dump(darkpaths, pkl)

        self.darkpaths = darkpaths

    def make_flat(self, savedir=None, do_bias=True, do_dark=True,
                  mbiaspath=None, mdarkpath=None,
                  comb_kwargs=MEDCOMB_KEYS, delimiter='-', dtype='float32'):
        '''Makes and saves flat images.
        Parameters
        ----------
        savedir: path-like, optional
            The directory where the frames will be saved.

        do_bias, do_dark : bool, optional
            If ``True``, subtracts bias and dark frames using
            ``self.biaspahts`` and ``self.darkpaths``. You can also
            specify ``mbiaspath`` and/or ``mdarkpath`` to ignore those
            in ``self.``.

        mbiaspath, mdarkpath : None, path-like, optional
            If you want to force a certain bias or dark to be used, then
            you can specify its path here.

        comb_kwargs: dict or None, optional
            The parameters for ``combine_ccd``.

        delimiter : str, optional.
            The delimiter for the renaming.

        dtype : str or numpy.dtype object, optional.
            The data type you want for the final master bias frame. It
            is recommended to use ``float32`` or ``int16`` if there is
            no specific reason.
        '''
        # Initial settings
        self.initialize_self()

        if savedir is None:
            savedir = self.topdir

        yfu.mkdir(savedir)
        flatpaths = {}

        # For simplicity, crop the original data by type_key and
        # type_val first.
        st = self.summary_raw.copy()
        for k, v in zip(self.flat_type_key, self.flat_type_val):
            st = st[st[k] == v]

        # For grouping, use type_key + group_key. This is because (1) it
        # is not harmful cuz type_key will have unique column values as
        # ``st`` has already been cropped in above for loop (2) by doing
        # this we get more information from combining process because,
        # e.g., "images with ["OBJECT", "EXPTIME"] = ["dark", 1.0] are
        # loaded" will be printed rather than just "images with
        # ["EXPTIME"] = [1.0] are loaded".
        gs = st.groupby(self.flat_key)

        # Do flat combine:
        for flat_val, flat_group in gs:
            # set path to master bias
            if mbiaspath is not None:
                biaspath = mbiaspath
            elif do_bias:
                # corresponding key for biaspaths:
                corr_bias = tuple(self.bias_type_val)
                # if _group_key not empty, add appropriate ``group_val``:
                if self.bias_group_key:
                    corr_bias += tuple(flat_group[self.bias_group_key].iloc[0])
                # else: empty. path is fully specified by _type_val.
                biaspath = self.biaspaths[corr_bias]
                try:
                    biaspath = self.biaspaths[corr_bias]
                except KeyError:
                    biaspath = None
                    warn(f"Bias not available for {corr_bias}. "
                         + "Processing without bias.")

            # set path to master dark
            if mdarkpath is not None:
                darkpath = mdarkpath
            elif do_dark:
                # corresponding key for darkpaths:
                corr_dark = tuple(self.dark_type_val)
                # if _group_key not empty, add appropriate ``group_val``:
                if self.dark_group_key:
                    corr_dark += tuple(flat_group[self.dark_group_key].iloc[0])
                # else: empty. path is fully specified by _type_val.
                try:
                    darkpath = self.darkpaths[corr_dark]
                except (KeyError):
                    darkpath = None
                    warn(f"Dark not available for {corr_dark}. "
                         + "Processing without dark.")

            # Do BD preproc before combine
            flat_bd_paths = []
            for i, flat_row in flat_group.iterrows():
                flat_orig_path = Path(flat_row["file"])
                flat_bd_path = (flat_orig_path.parent
                                / (flat_orig_path.stem + "_BD.fits"))
                ccd = yfu.load_ccd(flat_orig_path, unit='adu')
                _ = yfu.bdf_process(ccd,
                                    output=flat_bd_path,
                                    mbiaspath=biaspath,
                                    mdarkpath=darkpath,
                                    dtype="int16",
                                    overwrite=True,
                                    unit=None)
                flat_bd_paths.append(flat_bd_path)

            if not isinstance(flat_val, tuple):
                flat_val = tuple([flat_val])
            fname = delimiter.join([str(x) for x in flat_val]) + ".fits"
            fpath = Path(savedir) / fname

            _ = yfu.combine_ccd(flat_bd_paths,
                                output=fpath,
                                dtype=dtype,
                                **comb_kwargs,
                                normalize_average=True,  # Since skyflat!!
                                type_key=self.flat_key,
                                type_val=flat_val)

            flatpaths[tuple(flat_val)] = fpath

        # Save list of file paths for future use.
        # It doesn't take much storage and easy to erase if you want.
        with open(self.listdir / 'flatpaths.list', 'w+') as ll:
            for p in list(flatpaths.values()):
                ll.write(f"{str(p)}\n")

        with open(self.listdir / 'flatpaths.pkl', 'wb') as pkl:
            pickle.dump(flatpaths, pkl)

        self.flatpaths = flatpaths

    def do_preproc(self, savedir=None, delimiter='-', dtype='float32',
                   mbiaspath=None, mdarkpath=None, mflatpath=None,
                   do_bias=True, do_dark=True, do_flat=True,
                   do_crrej=False, crrej_kwargs=None, verbose_crrej=False,
                   verbose_bdf=True, verbose_summary=False):
        ''' Conduct the preprocessing using simplified ``bdf_process``.
        Parameters
        ----------
        savedir: path-like, optional
            The directory where the frames will be saved.

        delimiter : str, optional.
            The delimiter for the renaming.

        dtype : str or numpy.dtype object, optional.
            The data type you want for the final master bias frame. It is
            recommended to use ``float32`` or ``int16`` if there is no
            specific reason.

        mbiaspath, mdarkpath, mflatpath : None, path-like, optional
            If you want to force a certain bias, dark, or flat to be used,
            then you can specify its path here.

        crrej_kwargs : dict or None, optional
            If ``None`` (default), uses some default values defined in
            ``~.misc.LACOSMIC_KEYS``. It is always discouraged to use
            default except for quick validity-checking, because even the
            official L.A. Cosmic codes in different versions (IRAF, IDL,
            Python, etc) have different default parameters, i.e., there is
            nothing which can be regarded as default.
            To see all possible keywords, do
            ``print(astroscrappy.detect_cosmics.__doc__)``
            Also refer to
            https://nbviewer.jupyter.org/github/ysbach/AO2019/blob/master/Notebooks/07-Cosmic_Ray_Rejection.ipynb
        '''
        # Initial settings
        self.initialize_self()

        if savedir is None:
            savedir = self.topdir

        savedir = Path(savedir)
        yfu.mkdir(savedir)

        savepaths = []
        # for i, row in self.summary_raw.iterrows():
        #     fpath = Path(row["file"])
        for fpath in self.objpaths:
            savepath = savedir / Path(fpath).name
            savepaths.append(savepath)
            row = self.summary_raw[self.summary_raw["file"].values == str(
                fpath)]

            if mbiaspath is not None:
                biaspath = mbiaspath
            elif do_bias:
                # corresponding key for biaspaths:
                corr_bias = tuple(self.bias_type_val)
                # if _group_key not empty, add appropriate ``group_val``:
                if self.bias_group_key:  # not empty
                    corr_bias += tuple(row[self.bias_group_key].iloc[0])
                # else: empty. path is fully specified by _type_val.
                try:
                    biaspath = self.biaspaths[corr_bias]
                except (KeyError):
                    biaspath = None
                    warn(f"Bias not available for {corr_bias}. "
                         + "Processing without bias.")

            if mdarkpath is not None:
                darkpath = mdarkpath
            elif do_dark:
                # corresponding key for darkpaths:
                corr_dark = tuple(self.dark_type_val)
                # if _group_key not empty, add appropriate ``group_val``:
                if self.dark_group_key:
                    corr_dark += tuple(row[self.dark_group_key].iloc[0])
                # else: empty. path is fully specified by _type_val.
                try:
                    darkpath = self.darkpaths[corr_dark]
                except (KeyError):
                    darkpath = None
                    warn(f"Dark not available for {corr_dark}. "
                         + "Processing without dark.")

            if mflatpath is not None:
                flatpath = mflatpath
            elif do_flat:
                # corresponding key for darkpaths:
                corr_flat = tuple(self.flat_type_val)
                # if _group_key not empty, add appropriate ``group_val``:
                if self.flat_group_key:
                    corr_flat += tuple(row[self.flat_group_key].iloc[0])
                # else: empty. path is fully specified by _type_val.
                try:
                    flatpath = self.flatpaths[corr_flat]
                except (KeyError):
                    flatpath = None
                    warn(f"Flat not available for {corr_flat}. "
                         + "Processing without flat.")

            objccd = CCDData.read(fpath)
            _ = yfu.bdf_process(objccd,                       #
                                output=savepath,              #
                                unit=None,                    #
                                mbiaspath=biaspath,           #
                                mdarkpath=darkpath,           #
                                mflatpath=flatpath,           #
                                do_crrej=do_crrej,            #
                                crrej_kwargs=crrej_kwargs,    #
                                verbose_crrej=verbose_crrej,  #
                                verbose_bdf=verbose_bdf)

        self.reducedpaths = savepaths
        self.summary_red = yfu.make_summary(
            self.reducedpaths,
            output=self.topdir / "summary_reduced.csv",
            pandas=True,
            keywords=self.summary_keywords + ["PROCESS"],
            verbose=verbose_summary
        )
        return self.summary_red

    def make_astrometry_script(self, output=Path("astrometry.sh"),
                               log=Path("astrometry.log"), 
                               indexdir=Path('.'), cfg=Path("astrometry.cfg")):

        # It just simply assumes 1-pixel binning (no binnig).
        self.initialize_self()

        str_time = (r'current_date_time="`date +%Y-%m-%d\ %H:%M:%S`";'
                    + 'echo $current_date_time;')
        str_mv = "mv {} {}/input.fits"
        str_wcs = ("solve-field {}/input.fits -N {}"
                   + " --sigma 5 --downsample 4"
                   + " --radius 0.2 -u app -L 0.30 -U 0.33"
                   + " --cpulimit 300 --no-plot --overwrite --no-remove-lines")
        if not Path(cfg).exists():
            warn(f"astrometry config not found at {cfg} you specified.\n"
                 + f"Making it at path {cfg} using "
                 + f"the index directory ({indexdir}) you specified.")
            str_cfg = """
# This is a config file for the Astrometry.net 'astrometry-engine'
# program - it contains information about where indices are stored,
# and "site policy" items.

# Check the indices in parallel?
#
# -if the indices you are using take less than 2 GB of space, and you have at least
#  as much physical memory as indices, then you want this enabled.
#
# -if you are using a 64-bit machine and you have enough physical memory to contain
#  the indices you are using, then you want this enabled.
# 
# -otherwise, leave it commented-out.

inparallel

# If no scale estimate is given, use these limits on field width.
# minwidth 0.1
# maxwidth 180

# If no depths are given, use these:
#depths 10 20 30 40 50 60 70 80 90 100

# Maximum CPU time to spend on a field, in seconds:
# default is 600 (ten minutes), which is probably way overkill.
cpulimit 300

# In which directories should we search for indices?
add_path {:s}

# Load any indices found in the directories listed above.
autoindex

## Or... explicitly list the indices to load.
#index index-219
#index index-218
#index index-217
#index index-216
#index index-215
#index index-214
#index index-213
#index index-212
#index index-211
#index index-210
#index index-209
#index index-208
#index index-207
#index index-206
#index index-205
#index index-204-00
#index index-204-01
#index index-204-02
#index index-204-03
#index index-204-04
#index index-204-05
#index index-204-06
#index index-204-07
#index index-204-08
#index index-204-09
#index index-204-10
#index index-204-11
#index index-203-00
#index index-203-01
#index index-203-02
#index index-203-03
#index index-203-04
#index index-203-05
#index index-203-06
#index index-203-07
#index index-203-08
#index index-203-09
#index index-203-10
#index index-203-11
#index index-202-00
#index index-202-01
#index index-202-02
#index index-202-03
#index index-202-04
#index index-202-05
#index index-202-06
#index index-202-07
#index index-202-08
#index index-202-09
#index index-202-10
#index index-202-11
#index index-201-00
#index index-201-01
#index index-201-02
#index index-201-03
#index index-201-04
#index index-201-05
#index index-201-06
#index index-201-07
#index index-201-08
#index index-201-09
#index index-201-10
#index index-201-11
#index index-200-00
#index index-200-01
#index index-200-02
#index index-200-03
#index index-200-04
#index index-200-05
#index index-200-06
#index index-200-07
#index index-200-08
#index index-200-09
#index index-200-10
#index index-200-11
"""

            with open(Path(cfg), 'w+') as ff:
                ff.write(str_cfg.format(str(indexdir.resolve())))

        with open(Path(output), "w+") as astrometry:
            astrometry.write(str_time)
            astrometry.write("\n")
            for fpath in self.summary_red["file"]:
                fpath = Path(fpath)
                fparent = fpath.parent
                astrometry.write(str_mv.format(fpath, fparent))
                astrometry.write("\n")
                astrometry.write(str_wcs.format(fparent, fpath))
                astrometry.write("\n")
                astrometry.write(str_time)
                astrometry.write("\n")
            astrometry.write(f"rm {fparent}/input.*\n")
            astrometry.write(str_time)

        # import os
        # import tqdm

        # for fpath in tqdm(self.summary_red["file"]):
