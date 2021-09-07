'''
---------------
Before running
---------------

1.
Input file:  '*.fits'          Preprocessed FITS data
             '*.mag.1'         IRAF Phot file containing target's center info.
                               See below (i.e.,2. What you need to run this code)

Oupout file: 'sub_*.fits'      FITS data in which the subtraction technique is applied
                               so that the nearby stars and the background's gradient
                               are removed.

2.What you need to run this code.
  (The following packages must be installed.)
  i.  astropy (https://www.astropy.org/)
  ii. Source Extraction and Photometry (https://sep.readthedocs.io/en/v1.1.x/index.html)
  iii.``*.mag.1`` file from IRAF's Phot package that contains the center of the target (2005 UD).
      The first line should be the center of the target in the ordinary component, and the
      second line should be for the target in extraordinary component. We've also uploaded
      the ``*.mag.1`` file that we used with this code.

3.
In this code, the center of the target is found by using the phot task of IRAF.
So, we need the ``.mag`` file to import the coordinate of target's ceter.
There is no problem if you find the target's center by other methods.
All you need to do is modifying the part that brings the central coordinate of target.
See ``BRING THE CENTER COORDINATE OF 2005 UD`` part.
'''


import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime

from astropy.io import fits, ascii
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from astroquery.gaia import Gaia
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import sep


def pill_masking(image, x1, x2, y1, y2, height):
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


def circle_masking(image, x1, y1, r):

    Masking_image = np.zeros(np.shape(image))
    for yi in range(len(image)):
        for xi in range(len(image)):
            if (xi - x1)**2 + (yi-y1)**2 < (r)**2:
                Masking_image[yi, xi] = 1
    return Masking_image


def shift(image, shift_x, shift_y):
    '''
    Shift the image by (x+shift_x, y+shift_y)
    '''
    temp = np.roll(image, shift_x, axis=1)
    temp = np.roll(temp, shift_y, axis=0)
    return temp


def rround(x):
    '''Function to round a given number or list '''
    new_x = []
    if type(x) == 'int' or 'float':
        x = round(x)
        return(int(x))
    for i in range(len(x)):
        x_ = round(x[i])
        new_x.append(int(x_))
    return(np.array(new_x))

# ********************************************************************************************************** #
# *                                      BRING THE IMAGES                                                  * #
# ********************************************************************************************************** #


path = 'Directory/Where/processed/image/to/be/analyzed/are/saved/'  # Path of directory where images to be analyzed are saved
IMAGE_list = glob.glob(os.path.join(path, '*.fits'))  # Bring all FITS preprocessed" image in the directory
IMAGE_list = sorted(IMAGE_list)

# Bring the two images to do the image subtraction technique
for i in range(len(IMAGE_list)):
    image1 = IMAGE_list[i]  # image 1 (= to which the subtraction technique will be applied)

    if i == len(IMAGE_list)-1:
        image2 = IMAGE_list[i-1]  # image 2 (if image1 is the last image, image taken "before" image 1 is used.)
    else:
        image2 = IMAGE_list[i+1]  # image 2 (Image taken consecutively "after" image 1)

    # Open the image 1
    hdul1 = fits.open(image1)[0]
    data1 = hdul1.data
    header1 = hdul1.header
    EXPTIME = header1['EXPTIME'] / (24*60*60)  # in jd
    EXP_str = Time(header1['DATE-OBS'], format='isot').jd
    EXP_end = EXP_str + EXPTIME

    # Open the image 2
    hdul2 = fits.open(image2)[0]
    header2 = hdul2.header
    EXP_str2 = Time(header2['DATE-OBS'], format='isot').jd


# ********************************************************************************************************** #
# *                         BRING THE CENTER COORDINATE OF 2005 UD                                         * #
# ********************************************************************************************************** #
    # Here, we use Phot package in IRAF to find the center of the target
    # If you find the center of the target by other methods, please change this part.

    # Bring the center of iamge 1
    mag1 = ascii.read(image1+'.mag.1')
    xo1, yo1 = (mag1['XCENTER'][0]-1, mag1['YCENTER'][0]-1)  # pixel coordinate (x,y) of target's center in ordinary component
    xe1, ye1 = (mag1['XCENTER'][1]-1, mag1['YCENTER'][1]-1)  # pixel coordinate (x,y) of target's center in extra. component

    # Bring the center of iamge 2
    mag2 = ascii.read(image2 + '.mag.1')
    xo2, yo2 = (mag2['XCENTER'][0]-1, mag2['YCENTER'][0]-1)  # pixel coordinate (x,y) of target's center in ordinary component
    xe2, ye2 = (mag2['XCENTER'][1]-1, mag2['YCENTER'][1]-1)  # pixel coordinate (x,y) of target's center in extra. component


# ********************************************************************************************************** #
# *                        FIND THE NEARBY STARS AND THEIR PIXEL COORDINATE                                * #
# ********************************************************************************************************** #
    '''
    By using Gaia catalog, we find the nearby stars and convert their (ra,dec) coordinate to (x,y) (i.e., pixel coordiante).
    Due to the non-sidreal tracking, (x,y) of nearby stars at the exposure start point and the end point are different.
    So, we obtain (x,y) of nearby stars for both the exposure start point and end point.
    As a result, in one image, we have 4 pixel coordinate for one background star that is
    (x_star_str_o, y_star_str_o) : Pixel coordinate of a nearby star in ordinary component when the exposure starts
    (x_star_str_e, y_star_str_e) : Pixel coordinate of a nearby star in extra-ordinary component when the exposure ends
    (x_end_str_o, y_end_str_o)   : Pixel coordinate of a nearby star in ordinary component when the exposure starts
    (x_end_str_e, y_end_str_e)   : Pixel coordinate of a nearby star in extra-ordinary component when the exposure ends
    '''

    # Bring the (ra,dec) of 2005 UD when the exposure starts
    obj = Horizons(id=155140, location='Z23', epochs=EXP_str)
    eph = obj.ephemerides()
    ra_str, dec_str = eph['RA'][0], eph['DEC'][0]

    # Find the background stars nearby 2005UD by using Gaia
    coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.degree, u.degree), frame='icrs')
    width = u.Quantity(0.04, u.deg)  # Nearby stars within 0.04deg from the target
    height = u.Quantity(0.04, u.deg)
    r = Gaia.query_object_async(coordinate=coord, width=width, height=height)

    # (ra,dec) of nearby stars
    RA_star = []
    DEC_star = []
    g_star = []
    for i in range(len(r)):
        RA_star.append(r[i]['ra'])
        DEC_star.append(r[i]['dec'])
        g_star.append(r[i]['phot_g_mean_mag'])
    if len(g_star) == 0:
        print(image1 + ' has no nearby star')
        continue

    ######################
    #      IMAGE 1       #
    ######################
    # convert (ra,dec) to (x_star_str_o, y_star_str_o) of nearby stars in image 1
    header1['CRPIX1'], header1['CRPIX2'] = xo1, yo1
    header1['CRVAL1'], header1['CRVAL2'] = ra_str, dec_str
    w = WCS(header1)
    x_star_str_o1, y_star_str_o1 = w.wcs_world2pix(RA_star, DEC_star, 0)

    # convert (ra,dec) to (x_star_str_e, y_star_str_e) of nearby stars in image 1
    header1['CRPIX1'], header1['CRPIX2'] = xe1, ye1
    header1['CRVAL1'], header1['CRVAL2'] = ra_str, dec_str
    w = WCS(header1)
    x_star_str_e1, y_star_str_e1 = w.wcs_world2pix(RA_star, DEC_star, 0)

    # Bring the (ra,dec) of 2005 UD when the exposure ends
    obj = Horizons(id=155140, location='Z23', epochs=EXP_end)
    eph = obj.ephemerides()
    ra_end, dec_end = eph['RA'][0], eph['DEC'][0]

    # convert (ra,dec) to (x_star_end_o, y_star_end_o) of nearby stars in image 1
    header1['CRPIX1'], header1['CRPIX2'] = xo1, yo1
    header1['CRVAL1'], header1['CRVAL2'] = ra_end, dec_end
    w = WCS(header1)
    x_star_end_o1, y_star_end_o1 = w.wcs_world2pix(RA_star, DEC_star, 0)

    # convert (ra,dec) to (x_star_end_e, y_star_end_e) of nearby stars in image 1
    header1['CRPIX1'], header1['CRPIX2'] = xe1, ye1
    header1['CRVAL1'], header1['CRVAL2'] = ra_end, dec_end
    w = WCS(header1)
    x_star_end_e1, y_star_end_e1 = w.wcs_world2pix(RA_star, DEC_star, 0)

    ######################
    #      IMAGE 2       #
    ######################
    hdul2 = fits.open(image2)[0]
    data2 = hdul2.data
    header2 = hdul2.header
    EXPTIME2 = header2['EXPTIME']/(24*60*60)
    EXP_str2 = Time(header2['DATE-OBS'], format='isot').jd
    EXP_end2 = EXP_str2 + EXPTIME2

    # Bring the (ra,dec) of 2005 UD in image 2 when the exposure starts
    obj = Horizons(id=155140, location='Z23', epochs=EXP_str2)
    eph = obj.ephemerides()
    ra_str2, dec_str2 = eph['RA'][0], eph['DEC'][0]

    # convert (ra,dec) to (x_star_str_o, y_star_str_o) of nearby stars in image 2
    header2['CRPIX1'], header2['CRPIX2'] = xo2, yo2
    header2['CRVAL1'], header2['CRVAL2'] = ra_str2, dec_str2
    w = WCS(header2)
    x_star_str_o2, y_star_str_o2 = w.wcs_world2pix(RA_star, DEC_star, 0)

    # convert (ra,dec) to (x_star_str_e, y_star_str_e) of nearby stars in image 2
    header2['CRPIX1'], header2['CRPIX2'] = xe2, ye2
    header2['CRVAL1'], header2['CRVAL2'] = ra_str2, dec_str2
    w = WCS(header2)
    x_star_str_e2, y_star_str_e2 = w.wcs_world2pix(RA_star, DEC_star, 0)

    # Bring the (ra,dec) of 2005 UD in image 2 when the exposure ends
    obj = Horizons(id=155140, location='Z23', epochs=EXP_end2)
    eph = obj.ephemerides()
    ra_end2, dec_end2 = eph['RA'][0], eph['DEC'][0]

    # convert (ra,dec) to (x_star_end_o, y_star_end_o) of nearby stars in image 2
    header2['CRPIX1'], header2['CRPIX2'] = xo2, yo2
    header2['CRVAL1'], header2['CRVAL2'] = ra_end2, dec_end2
    w = WCS(header2)
    x_star_end_o2, y_star_end_o2 = w.wcs_world2pix(RA_star, DEC_star, 0)

    # convert (ra,dec) to (x_star_end_e, y_star_end_e) of nearby stars in image 2
    header2['CRPIX1'], header2['CRPIX2'] = xe2, ye2
    header2['CRVAL1'], header2['CRVAL2'] = ra_end2, dec_end2
    w = WCS(header2)
    x_star_end_e2, y_star_end_e2 = w.wcs_world2pix(RA_star, DEC_star, 0)

# ********************************************************************************************************** #
# *                                      MAKING THE MASKING IMAGE                                          * #
# ********************************************************************************************************** #
    '''
    Since the stars in the background are elongated, we mask them with with a pill box here.
    The width of pill box is determined by a rule of thumb.
    '''

    Masking_image_o1 = pill_masking(data1, x_star_str_o1, x_star_end_o1, y_star_str_o1, y_star_end_o1, 3)
    Masking_image_e1 = pill_masking(data1, x_star_str_e1, x_star_end_e1, y_star_str_e1, y_star_end_e1, 3)
    Masking_target_o1 = circle_masking(data1, xo1, yo1, 6)
    Masking_target_e1 = circle_masking(data1, xe1, ye1, 6)
    Masking_image1 = Masking_image_o1 + Masking_image_e1 + Masking_target_o1 + Masking_target_e1
    masked_image1 = np.ma.masked_array(data1, Masking_image1)

    Masking_image_o2 = pill_masking(data2, x_star_str_o2, x_star_end_o2, y_star_str_o2, y_star_end_o2, 3)
    Masking_image_e2 = pill_masking(data2, x_star_str_e2, x_star_end_e2, y_star_str_e2, y_star_end_e2, 3)
    Masking_target_o2 = circle_masking(data2, xo2, yo2, 6)
    Masking_target_e2 = circle_masking(data2, xe2, ye2, 6)
    Masking_image2 = Masking_image_o2 + Masking_image_e2 + Masking_target_o2 + Masking_target_e2
    masked_image2 = np.ma.masked_array(data2, Masking_image2)


# ********************************************************************************************************** #
# *                                      Derive Background value                                           * #
# ********************************************************************************************************** #

    # Derive median value of backbround in image1
    area_y1 = rround(yo1 - 15)
    area_y2 = rround(yo1 + 15)
    area_x1 = rround(xo1 - 10)
    area_x2 = rround(xo1 + 30)
    sky_1 = np.ma.median(masked_image1[area_y1:area_y2, area_x1:area_x2])  # Median value of sky #mask the background stars

    # Derive median value of backbround in image2
    area_y1 = rround(yo2 - 15)
    area_y2 = rround(yo2 + 15)
    area_x1 = rround(xo2 - 10)
    area_x2 = rround(xo2 + 30)
    sky_2 = np.ma.median(masked_image2[area_y1:area_y2, area_x1:area_x2])  # Median value of sky #mask the background stars


# ********************************************************************************************************** #
# *                         Making the image for subtraction technique                                     * #
# ********************************************************************************************************** #
    '''
    We modify the image2 to be applied for image1. To do this, the following work is performed.
    1. Remove 2005 UD in image2 and leave only the nearby stars.
    2. Shift the image2 so that background stars in image2 has the same pixel coordinate with those of in image 1.
    3. Adjust the backgound's offset between image 1 and image 2.
       At this time, polarization of the sky and background stars is not considered.
    '''
    #######################################################################
    #    1. Remove 2005 UD in image2 and leave only the nearby stars.     #
    #######################################################################

    data3 = np.copy(data2)  # Image will be used for the subtraction of image1
    data3 = data3.byteswap().newbyteorder()

    bkg = sep.Background(data3, mask=Masking_image2,
                         bw=3, bh=3, fw=3, fh=3)  # Background mapping #Here, values are empirically decided.
    bkg_image = bkg.back()

    Masking_target = Masking_target_o2 + Masking_target_e2  # Masking area of 2005UD
    Masking_target = (Masking_target).astype(bool)
    data3[Masking_target] = bkg_image[Masking_target]  # Overlay 2005UD with the background mapping image

    #######################################################################
    #                         2. Shift the image2                         #
    #######################################################################
    diff_star = np.sqrt((x_star_str_o1 - xo1)**2 + (y_star_str_o1 - yo1)**2)
    order = list(diff_star).index(min(diff_star))  # Select the closest nearby star from 2005 UD

    shift_x = rround(x_star_str_o1[order]-x_star_str_o2[order])
    shift_y = rround(y_star_str_o1[order]-y_star_str_o2[order])  # Find how much to shift image 2 based on the selected star

    #######################################################################
    #    3. Adjust the backgound's offset between image 1 and image 2     #
    #######################################################################

    # Derive median value of backbround in image1
    area_y1 = rround(yo1 - 15)
    area_y2 = rround(yo1 + 15)
    area_x1 = rround(xo1 - 10)
    area_x2 = rround(xo1 + 30)
    sky_1 = np.ma.median(masked_image1[area_y1:area_y2, area_x1:area_x2])  # Median value of sky #mask the background stars

    # Derive median value of backbround in image2
    area_y1 = rround(yo2 - 15)
    area_y2 = rround(yo2 + 15)
    area_x1 = rround(xo2 - 10)
    area_x2 = rround(xo2 + 30)
    sky_2 = np.ma.median(masked_image2[area_y1:area_y2, area_x1:area_x2])  # Median value of sky #mask the background stars

    # MAKE THE SUBTRACTED IMAGE TO BE APPLIED ON IMAGE 1 BY USING IMAGE 2
    subtracted_image = shift(data3, shift_x, shift_y)*sky_1/sky_2


# ********************************************************************************************************** #
# *                    FINAL IMAGE AFTER APPLIED THE SUBTRACTION TECHNIQUE                                 * #
# ********************************************************************************************************** #
    final_image = data1 - subtracted_image  # FINAL IMAGE AFTER APPLIED THE SUBTRACTION TECHNIQUE

    FILENAME = os.path.join(path, 'sub_'+image1.split('/')[-1])
    header1['COMMENT'] = 'Background is subtracted by ' + image2 + datetime.now().strftime('%Y-%m-%d %H:%M:%S(KST)')
    fits.writeto(FILENAME, final_image, header=header1, overwrite=True )  # Save it as FITS file

    # Plot the image 1 and image after the subtraction
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    im = ax[0].imshow(data1, vmin=sky_1*0.7, vmax=sky_1*1.3)
    ax[0].set_xlim(100, 300)
    ax[0].set_ylim(100, 300)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    gap = sky_1*0.6
    im = ax[1].imshow(final_image, vmin=-gap, vmax=+gap, cmap='seismic')
    ax[1].set_xlim(100, 300)
    ax[1].set_ylim(100, 300)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.show()
