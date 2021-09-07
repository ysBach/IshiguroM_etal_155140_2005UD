'''
---------------
Before running
---------------

This is the code for making the "Masking image" of FITS file.
The "Masking image" masks the 1) nearby stars, 2) Cosmic ray, (if this is MSI data) 3) the polarization mask area.

1.
Input file:  '*.fits'                      Preprocessed FITS file
             '*.mag.1'                     IRAF Phot file containing target's center info.
                                           See below (i.e.,2. What you need to run this code)
             'dome_impol_*.fits'           Dome flat FITS file  (ONLY FOR MSI DATA)
                                           (in the case of NOT data, you don't need it)

Oupout file: 'mask_*.fits'                 Masking image in FITS format

2.What you need to run this code.
  (The following packages must be installed.)
  i.  astropy (https://www.astropy.org/)
  ii. Astro-SCRAPPY (https://github.com/astropy/astroscrappy)
  iii.``*.mag.1`` file from IRAF's Phot package that contains the center of the target (2005 UD).
      The first line should be the center in the ordinary component, and the
      second line should be for extraordinary component.
      We've also uploaded the ``*.mag.1`` file that we used with this code.


3.
In this code, the center of the target is found by using the phot of IRAF.
So, we need the ``.mag`` file to bring the coordinate of target's ceter.
There is no problem if you find the target's center by other methods.
All you need to do is modifying the part that brings the central coordinate of target.
See ``Bring Pixel coordinate of target`` part.

4.
Directory should contain the complete sets consist of 4 images (taken at HWP=0, 22.5, 45, 67.5 deg).
If even one set does not have 4 images (e.g., set having images taken at HWP = 0, 45, 67.5 deg),
an error will occur.
'''

import glob
import os
import numpy as np
import warnings
import pandas as pd

from astropy.io import fits, ascii
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from astroquery.gaia import Gaia
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning
import astroscrappy
warnings.simplefilter('ignore', category=AstropyWarning)


# ********************************************************************************************************** #
# *                                           INPUT VALUE                                                  * #
# ********************************************************************************************************** #

INSTRUMENT = 'MSI'  # Choose which data you use. one of 'MSI' or 'NOT'
Observatory = 'q33'  # Nayoro observatory: 'q33', NOT: 'Z23'
OBJECT = 155140  # 2005 UD
Limiting_mag = 18  # Masking the star brighter than the given magnitude
Pill_masking_with = 20  # [pix] #The Width of pill box masking the nearby stars

path = os.path.join('Directory/Where/image/to/be/masked/is/saved/')  # Path of directory where images to be masked are saved
IMAGE_list = glob.glob(os.path.join(path, '*.fits'))  # Bring all FITS preprocessed image to be masked in the directory
# ********************************************************************************************************** #


IMAGE_list = sorted(IMAGE_list)
order = np.arange(0, len(IMAGE_list), 4)
for z in order:
    SET = [IMAGE_list[z], IMAGE_list[z+1], IMAGE_list[z+2], IMAGE_list[z+3]]

    for i in range(0, 4):
        # ********************************************************************************************************** #
        # *                          BRING THE IMAGES TO BE MASKED                                                 * #
        # ********************************************************************************************************** #
        RET = SET[i]  # Bring the fits file
        hdul = fits.open(RET)[0]
        header = hdul.header
        image = hdul.data
        exptime = header['EXPTIME']  # in sec

        Mask_image = np.zeros(np.shape(image))

        # --------------------------------------------------
        # Read Header info. needed for astroquery.horizons
        # --------------------------------------------------
        if INSTRUMENT == 'MSI':
            EXP_str = header['MJD-STR'] + 2400000.5
            EXP_end = header['MJD-END'] + 2400000.5
            RET_ANG2 = header['RET-ANG2']
            width_ = 0.054
            height_ = 0.009
        elif INSTRUMENT == 'NOT':
            EXP_str = Time(header['DATE-OBS'], format='isot').jd
            EXP_end = EXP_str + exptime / (24*60*60)
            RET_ANG2 = header['FARETANG']
            width_ = 0.05
            height_ = 0.05

        # --------------------------------------------------
        # Bring the observer quantities from JPL Horizons
        # --------------------------------------------------
        # Querying the RA, DEC of target
        obj = Horizons(id=OBJECT, location=Observatory, epochs=EXP_str)
        eph = obj.ephemerides()
        ra, dec = eph['RA'][0], eph['DEC'][0]

        # --------------------------------------------------
        # Find the nearby star RA,DEC from GAIA
        # --------------------------------------------------
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
        width = u.Quantity(width_, u.deg)
        height = u.Quantity(height_, u.deg)
        r = Gaia.query_object_async(coordinate=coord, width=width, height=height)

        star_list = pd.DataFrame({'RA': [],
                                  'DEC': [],
                                  'Mag': []})

        for i in range(len(r)):
            star_list = star_list.append({'RA': r[i]['ra'],
                                          'DEC': r[i]['dec'],
                                          'Mag': r[i]['phot_g_mean_mag']},
                                         ignore_index=True)

        star_list = star_list[star_list['Mag'] < Limiting_mag]

        # --------------------------------------------------
        # Bring Pixel coordinate of target
        # --------------------------------------------------
        # Here, we use Phot package in IRAF to find the center of the target
        # If you find the center of the target by other methods, please change this part.
        data = ascii.read(RET+'.mag.1')
        xo, yo = data['XCENTER'][0]-1, data['YCENTER'][0]-1  # Center of ordinary light
        xe, ye = data['XCENTER'][1]-1, data['YCENTER'][1]-1  # "  of extraordinary light

        # --------------------------------------------------
        # Masking nearby star
        # --------------------------------------------------
        def pill_masking(image, x1, x2, y1, y2, height, target_x, target_y, target_radi=8):
            x_star_str = x1
            x_star_end = x2
            y_star_str = y1
            y_star_end = y2

            Masking_image = np.zeros(np.shape(image))
            for yi in range(len(image)):
                for xi in range(len(image)):
                    for star in range(len(x_star_end)):
                        slope = (y_star_end[star] - y_star_str[star])/(x_star_end[star]-x_star_str[star])
                        y_up = slope * xi + y_star_str[star] + height - slope * x_star_str[star]
                        y_low = slope * xi + y_star_str[star] - height - slope * x_star_str[star]
                        x_str = min(x_star_str[star], x_star_end[star])
                        x_end = max(x_star_str[star], x_star_end[star])

                        if (xi - x_star_str[star])**2 + (yi-y_star_str[star])**2 < (height)**2:
                            Masking_image[yi, xi] = 1
                        if (xi - x_star_end[star])**2 + (yi-y_star_end[star])**2 < (height)**2:
                            Masking_image[yi, xi] = 1
                        if yi >= y_low and y_up >= yi and xi > x_str and x_end > xi:
                            Masking_image[yi, xi] = 1
            return Masking_image

        # convert star's (ra,dec) to (x_star_str_o, y_star_str_o) of nearby stars
        header['CRVAL1'], header['CRVAL2'] = ra, dec
        header['CRPIX1'], header['CRPIX2'] = xo, yo
        w = WCS(header)
        x_star_str_o, y_star_str_o = w.wcs_world2pix(star_list['RA'].values,
                                                     star_list['DEC'].values, 0)

        # convert star's (ra,dec) to (x_star_str_e, y_star_str_e) of nearby stars
        header['CRVAL1'], header['CRVAL2'] = ra, dec
        header['CRPIX1'], header['CRPIX2'] = xe, ye
        w = WCS(header)
        x_star_str_e, y_star_str_e = w.wcs_world2pix(star_list['RA'].values,
                                                     star_list['DEC'].values, 0)

        # Bring (ra,dec) of target at the end of exposure
        obj = Horizons(id=OBJECT, location=Observatory, epochs=EXP_end)
        eph = obj.ephemerides()
        ra, dec = eph['RA'][0], eph['DEC'][0]

        # convert star's (ra,dec) to (x_star_str_o, y_star_str_o) of nearby stars
        header['CRVAL1'], header['CRVAL2'] = ra, dec
        header['CRPIX1'], header['CRPIX2'] = xo, yo
        w = WCS(header)
        x_star_end_o, y_star_end_o = w.wcs_world2pix(star_list['RA'].values,
                                                     star_list['DEC'].values, 0)

        # convert star's (ra,dec) to (x_star_str_e, y_star_str_e) of nearby stars
        header['CRVAL1'], header['CRVAL2'] = ra, dec
        header['CRPIX1'], header['CRPIX2'] = xe, ye
        w = WCS(header)
        x_star_end_e, y_star_end_e = w.wcs_world2pix(star_list['RA'].values,
                                                     star_list['DEC'].values, 0)

        Masking_image_o = pill_masking(image, x_star_str_o, x_star_end_o, y_star_str_o, y_star_end_o,
                                       Pill_masking_with/2, xo, yo, Pill_masking_with/2)
        Masking_image_e = pill_masking(image, x_star_str_e, x_star_end_e, y_star_str_e, y_star_end_e,
                                       Pill_masking_with/2, xe, ye, Pill_masking_with/2)
        # Here, the width of pill box for masking is determined empirically.
        # The width could be improved by specifying depending on the star's brightness in future work.

        Mask_image = Masking_image_o  + Masking_image_e

        # --------------------------------------------------
        # Mask the polarization mask area. (Only for MSI data)
        # --------------------------------------------------
        if INSTRUMENT == 'MSI':
            FLAT = os.path.join(path, 'Flat',
                                'dome_impol_r_hwp{:04.1f}*'.format(RET_ANG2))
            # Bring dome flat image taken at HWP angle
            FLAT = glob.glob(FLAT)[0]
            flat = fits.open(FLAT)[0]
            flat0 = flat.data
            if flat0[300, 250] < 1:
                Mask_image[flat0 < 0.85] = 1  # Masking the polarization masked area
            elif flat0[300, 250] > 1:
                Mask_image[flat0 < 2.4] = 1
            Mask_image[flat0 == 1] = 1   # Masking the polarization masked area

        # --------------------------------------------------
        # Mask the cosmic-ray
        # --------------------------------------------------
        gain = header['GAIN']
        m_LA, cor_image = astroscrappy.detect_cosmics(image,
                                                      sepmed=False,
                                                      gain=gain,
                                                      readnoise=4.5,
                                                      sigclip=8.5)  # This value can be changed.
        tmLA = m_LA.astype(int)
        Mask_image[tmLA == 1 ] = 1

        # --------------------------------------------------
        # Make the FITS file of the masking image
        # --------------------------------------------------
        if not os.path.exists(os.path.join(path, 'Masking')):
            os.makedirs(os.path.join(path, 'Masking'))
        mask_path = os.path.join(path, 'Masking', 'mask_'+RET.split('/')[-1])
        fits.writeto(mask_path, data=Mask_image, header=header, overwrite=True)
        print(mask_path + ' is created.')
