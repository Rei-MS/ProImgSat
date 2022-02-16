import os
import sys
import math
import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from matplotlib import image as img
from matplotlib import patches as pat

# Turn interactive mode on for pyplot.
plt.ion()


###
# Read all bands (1 through 7, without 6) from image LandsatTM229_82 and store
# them in numpy arrays.
###


# Get current working directory.
default_dir = os.getcwd()

# Set directory to image location, change as necessary.
os.chdir(r"C:\Users\Reinaldo\Desktop\Facu\ProImgSat\Practico 3\LandsatTM229_82")

# Read the header.
with open('header.dat') as file:
    header = file.read()
#print(header)

# Now we need to get data from the header. We need the size of the bands.
# To get the rows and columns we get data from the header.
# Since we have the string for the header, we can read the numbers, but
# we might aswell try to get them from the string. We know they're both
# 4 digit numbers.
row_string = 'LINES PER IMAGE= '
col_string = 'PIXELS PER LINE= '

# Slice the header string to get the desired numbers
header_row_index = header.find(row_string) + len(row_string)
rows = int(header[header_row_index : header_row_index + 4 ]) + 1
#print(rows)
# Apparently, header data is wrong, so 1 is added.

header_col_index = header.find(col_string) + len(col_string)
columns = int(header[header_col_index : header_col_index + 4])
#print(columns)

# Initialize array to store the image. Since we are storing an image, it's
# data type should be set to 'np.uint8', that is an unsigned integer between
# 0 and 255. Here, 6 is the number of bands.
image = np.zeros((rows, columns, 6), dtype = np.uint8)

# Now, fill the array with the corresponding data. We are storing each
# band array in a dictionary. This is to avoid confusion over indexing
# when referring to a specific band. Instead of using image[:, :, 5] for
# band number 7, band['7w'] will be used. 'w' is for whole.
band = {}
for bands in [1, 2, 3, 4, 5, 7]:
    with open('BAND' + str(bands) + '.dat','rb') as band_file:
        band_flat = np.fromfile(band_file, dtype = np.uint8)
        try:
            image[:, :, bands - 1] = band_flat.reshape(rows, columns)
            band['{}w'.format(bands)] = image[:, :, bands - 1]
        except IndexError:
            image[:, :, bands - 2] = band_flat.reshape(rows, columns)
            band['{}w'.format(bands)] = image[:, :, bands - 2]

# Return to default directory. This is to avoid unnecessary conflicts.
os.chdir(default_dir)

print(' ')
# Ask whether to show a specific band or not.
show_a_band = input('Show a specific band as an image? (y/n): ')

if show_a_band == str('y'):
    show_band = input('Which band? (1, 2, 3, 4, 5 or 7): ')
    try:
        if int(show_band) in [1, 2, 3, 4, 5, 7]:
            plt.figure(0, frameon=False)
            plt.imshow(band['{}w'.format(show_band)],"gray")
            plt.title("Image for band number " + show_band + '.')
            plt.show()
        else:
            sys.exit('Script stopped. Must input 1, 2, 3, 4, 5 or 7.')
    except ValueError:
        sys.exit('Script stopped. Input was not an integer.')
elif show_a_band == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Crop all bands into 2000 by 2000 arrays that show Cordoba and
# Carlos Paz. Cropped bands will be stored in dictionary.
# Normalization will also be done.
###


for bands in [1, 2, 3, 4, 5, 7]:
    # Normalize.
    band['{}w'.format(bands)] = band['{}w'.format(bands)]/255
    # Crop.
    band['{}'.format(bands)] = band['{}w'.format(bands)][1500:3500, 3150:5150]


print(' ')
# Ask whether to show a specific cropped band or not.
show_ac_band = input('Show a specific cropped band as an image? (y/n): ')

if show_ac_band == str('y'):
    show_c_band = input('Which band? (1, 2, 3, 4, 5 or 7): ')
    try:
        if int(show_c_band) in [1, 2, 3, 4, 5, 7]:
            plt.figure(0, frameon=False)
            plt.imshow(band['{}'.format(show_c_band)], "gray")
            plt.title("Image for cropped band number " + show_c_band + '.')
            plt.show()
        else:
            sys.exit('Script stopped. Must input 1, 2, 3, 4, 5 or 7.')
    except ValueError:
        sys.exit('Script stopped. Input was not an integer.')
elif show_ac_band == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Display bands 1, 2, 3 and 4, 5, 7 as an image.
###

print(' ')
# Ask whether to show these two images or not.
show_images = input('Show bands 1, 2, 3 and 4, 5, 7 as images? (y/n): ')

if show_images == str('y'):
    image_shower = np.zeros((2000, 2000, 3))
    for index, bands in enumerate([3, 2, 1]):
        image_shower[:, :, index] = band['{}'.format(bands)]

    plt.figure(frameon=False)
    plt.title('Bands 321 as an image.')
    plt.imshow(image_shower)
    plt.show()

    for index, bands in enumerate([7, 5, 4]):
        image_shower[:, :, index] = band['{}'.format(bands)]

    plt.figure(frameon=False)
    plt.title('Bands 754 as an image.')
    plt.imshow(image_shower)
    plt.show()

elif show_images == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Plot a histogram for each band.
###


print(' ')
# Ask whether to pick bands and show their histograms or not.
show_histograms = input('Pick bands to plot their histograms? (y/n): ')

if show_histograms == str('y'):
    try:
        print('Input bands as a single number. (i.e., for the first three bands input 123): ')
        hist_string = int(input('- '))
    except ValueError:
        sys.exit('Script stopped. Input was not an integer.')

    checker = []
    for bands in range(len(str(hist_string))):
        if int(str(hist_string)[bands]) in [1, 2, 3, 4, 5, 7]:
            checker.append(str(hist_string)[bands])

        else:
            sys.exit('Script stopped. Invalid band in input.')

    if len(set(checker)) == len(checker):
        histogram_shower = []
        for index, bands in enumerate(checker):
            hi = ndi.histogram(band['{}'.format(bands)], 0, 1, 256)
            histogram_shower.append(hi)
            plt.figure(frameon=False)
            plt.title('Histogram. Band number ' + bands + '.')
            plt.plot(np.linspace(0, 1, 256), histogram_shower[index], 'g')
            plt.show()


    else:
        sys.exit('Script stopped. Band was repeated.')

elif show_histograms == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Define and apply a linear enhancement at 2%.
###

# First, make sure to have a histogram for each of the 6 bands.
histograms = {}

for bands in [1, 2, 3, 4, 5, 7]:
    histograms['{}'.format(bands)] = ndi.histogram(band['{}'.format(bands)], 0, 1, 256)


# Function - enhancement
# Returns a band mapped via a linear enhancement (percentile).
# Parameters:
#   v_band - the band one wishes to map.
#   histogram - the histogram of said band.
#   percentage - percentage.
def enhancement(v_band, histogram, percentage = 2):
    # Cumulative distribution function.
    cdf = np.cumsum(histogram)
    cdf_max = cdf[-1] * ((percentage / 2) / 100)
    cdf_min = cdf[-1] * ((100 - (percentage / 2)) / 100)

    # Calculate endpoints.
    try:
        endpoint_max = np.argwhere(cdf < cdf_max)[-1] / len(histogram)
        endpoint_min = np.argwhere(cdf > cdf_min)[0]  / len(histogram)
    except IndexError:
        if np.shape(np.argwhere(cdf < cdf_max))[0] == 0:
            endpoint_max = 1
            endpoint_min = np.argwhere(cdf > cdf_min)[0]  / len(histogram)
        else:
            endpoint_max = np.argwhere(cdf < cdf_max)[-1] / len(histogram)
            endpoint_min = 0

    # Apply mapping function.
    en_band = np.divide(v_band - endpoint_min, endpoint_max - endpoint_min)

    # Kill outliers.
    en_band[en_band < 0] = 0
    en_band[en_band > 1] = 1

    return en_band


# Apply enhancement.
for bands in [1, 2, 3, 4, 5, 7]:
    band['{}en'.format(bands)] = enhancement(band['{}'.format(bands)], histograms['{}'.format(bands)])
    histograms['{}en'.format(bands)] = ndi.histogram(band['{}en'.format(bands)], 0, 1, 256)


print(' ')
# Ask whether to pick bands and show their histograms before and after enhancement or not.
show_histogram_comparison = input('Pick bands to plot their histograms before and after enhancement? (y/n): ')

if show_histogram_comparison == str('y'):
    try:
        print('Input bands as a single number. (i.e., for the first three bands input 123): ')
        hist_string = int(input('- '))
    except ValueError:
        sys.exit('Script stopped. Input was not an integer.')

    checker = []
    for bands in range(len(str(hist_string))):
        if int(str(hist_string)[bands]) in [1, 2, 3, 4, 5, 7]:
            checker.append(str(hist_string)[bands])

        else:
            sys.exit('Script stopped. Invalid band in input.')

    if len(set(checker)) == len(checker):

        for index, bands in enumerate(checker):
            plt.figure(frameon=False, figsize=(10, 6.5))
            # Subplot for histogram before enhancement.
            plt.subplot(211)
            plt.title('Histogram before enhancement. Band number ' + bands + '.')
            plt.plot(np.linspace(0, 1, 256), histograms['{}'.format(bands)], 'g')
            # Subplot for histogram after enhancement.
            plt.subplot(212)
            plt.title('Histogram after enhancement. Band number ' + bands + '.')
            plt.plot(np.linspace(0, 1, 256), histograms['{}en'.format(bands)], 'g')

            plt.show()

    else:
        sys.exit('Script stopped. Band was repeated.')

elif show_histogram_comparison == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')

'''
Respuesta punto 5 (que va de la mano con el 4):
Los picos no cambian en el sentido de que el realce lo que hace es agarrar y 'estirar' el histograma para que cubra
todos los valores posibles (0-255).
'''


print(' ')
# Ask whether to show 3 different enhanced bands as an image or not. It gives
# the option of selecting triplets until told otherwise.
show_three_band = input('Pick an enhanced band triplet to show as an image? (y/n): ')

if show_three_band == str('y'):
    print('Order is important. Must be RGB, in that order. Must input 1, 2, 3, 4, 5 or 7.')
    print('No repeats.')
    print(' ')
    while True:
        image_shower = np.zeros((2000, 2000, 3), dtype = np.single)
        checker = []

        for i in range(3):
            checker.append(input(str(i+1) + ' - Input a band number: '))

        if len(set(checker)) == len(checker):
            try:
                for i in range(3):
                    image_shower[:, :, i] = band['{}en'.format(checker[i])]

                plt.figure(frameon=False)
                plt.title('Bands ' + str(checker[0]) + str(checker[1]) + str(checker[2]) + ' as an image.')
                plt.imshow(image_shower)
                plt.show()

            except KeyError:
                print('Invalid Input. Returning to selection.')
                print(' ')
                continue

            print(' ')
            another_one = input('Pick another triplet? (y/n): ')
            print(' ')

            if another_one == str('y'):
                continue

            elif another_one == str('n'):
                break

            else:
                print('Invalid input. Image showing stropped.')
                break

        else:
            print(' ')
            repeat = input('A band was repeated. Try again? (y/n): ')
            print(' ')

            if repeat == str('y'):
                continue

            elif repeat == str('n'):
                break

            else:
                print('Invalid input. Image showing stropped.')
                break

elif show_three_band == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Obtain histograms for land covers.
# Water, city, mountain, crops and vegetation are considered.
# Compare with the enhanced image histograms.
###

# Array to store enhanced bands.
image_shower = np.zeros((2000, 2000, 6))

# Store bands 457 and 312, in that order (i.e. band 1 is in index 4).
for index, bands in enumerate([4, 5, 7, 3, 1, 2]):
    image_shower[:, :, index] = band['{}en'.format(bands)]

# Slicing.
rec_water = image_shower[950:1050, 335:395, :]
rec_city = image_shower[950:1200, 1250:1550, :]
rec_mountain = image_shower[25:260, 505:725, :]
rec_crops = image_shower[150:230, 1450:1695, :]
rec_vegetation = image_shower[240:470, 1640:1695, :]


# Ask whether to show the land covers or not.
print(' ')
show_land_covers = input('Show land cover rectangles? (y/n): ')

if show_land_covers == str('y'):
    # Show areas selected for land covers.
    area_shower = np.zeros((2000, 2000, 3))
    for index, bands in enumerate([3, 2, 1]):
        area_shower[:, :, index] = band['{}'.format(bands)]

    # Create rectangles for each land cover.
    rect_water = pat.Rectangle((335, 950), 60, 100, linewidth=1, edgecolor='r', facecolor='none')
    rect_city = pat.Rectangle((1250, 950), 300, 250, linewidth=1, edgecolor='r', facecolor='none')
    rect_mountain = pat.Rectangle((505, 25), 220, 235, linewidth=1, edgecolor='r', facecolor='none')
    rect_crops = pat.Rectangle((1450, 150), 245, 80, linewidth=1, edgecolor='r', facecolor='none')
    rect_vegetation = pat.Rectangle((1640, 240), 55, 230, linewidth=1, edgecolor='r', facecolor='none')

    plt.figure(frameon=False)
    plt.title('Rectangles indicate land cover rectangles selected.')
    plt.imshow(area_shower)
    plt.gca().add_patch(rect_water)
    plt.gca().add_patch(rect_city)
    plt.gca().add_patch(rect_mountain)
    plt.gca().add_patch(rect_crops)
    plt.gca().add_patch(rect_vegetation)

    plt.show()

    # Show land covers for bands 457 and 312.
    # For RGB channels as bands 457.
    plt.figure(frameon=False)
    plt.suptitle('RGB channels as bands 457.')

    plt.subplot(231)
    plt.title('Water.')
    plt.imshow(rec_water[:, :, 0:3])

    plt.subplot(232)
    plt.title('Mountain.')
    plt.imshow(rec_mountain[:, :, 0:3])

    plt.subplot(233)
    plt.title('Vegetation.')
    plt.imshow(rec_vegetation[:, :, 0:3])

    plt.subplot(223)
    plt.title('City.')
    plt.imshow(rec_city[:, :, 0:3])

    plt.subplot(224)
    plt.title('Crops')
    plt.imshow(rec_crops[:, :, 0:3])

    plt.show()

    # For RGB channels as bands 312.
    plt.figure(frameon=False)
    plt.suptitle('RGB channels as bands 312.')

    plt.subplot(231)
    plt.title('Water.')
    plt.imshow(rec_water[:, :, 3:])

    plt.subplot(232)
    plt.title('Mountain.')
    plt.imshow(rec_mountain[:, :, 3:])

    plt.subplot(233)
    plt.title('Vegetation.')
    plt.imshow(rec_vegetation[:, :, 3:])

    plt.subplot(223)
    plt.title('City.')
    plt.imshow(rec_city[:, :, 3:])

    plt.subplot(224)
    plt.title('Crops')
    plt.imshow(rec_crops[:, :, 3:])

    plt.show()
elif show_land_covers == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


# Ask whether to show histogram comparison or not.
print(' ')
print('WARNING: The following step will display 10 figures.')
show_lc_hist = input('Show comparison histograms for land covers? (y/n): ')

if show_lc_hist == str('y'):
    land_covers = ['water', 'city', 'mountain', 'crops', 'vegetation']
    land_covers_images = [rec_water, rec_city, rec_mountain, rec_crops, rec_vegetation]

    for index, cover in enumerate(land_covers):
        # RGB channels as bands 457.
        plt.figure(frameon=False)
        plt.suptitle('RGB channels as bands 457. Left: ' + str(cover) + '. Right: Original image.')

        # R channel for enhanced image.
        plt.subplot(322)
        plt.plot(np.linspace(0, 1, 256), histograms['4en'], 'r')

        # G channel for enhanced image.
        plt.subplot(324)
        plt.plot(np.linspace(0, 1, 256), histograms['5en'], 'g')

        # B channel for enhanced image.
        plt.subplot(326)
        plt.plot(np.linspace(0, 1, 256), histograms['7en'], 'b')

        # R channel land cover.
        plt.subplot(321)
        r_hist_321 = ndi.histogram(land_covers_images[index][:, :, 0], 0, 1, 256)
        plt.plot(np.linspace(0, 1, 256), r_hist_321, 'r')

        # G channel land cover.
        plt.subplot(323)
        g_hist_321 = ndi.histogram(land_covers_images[index][:, :, 1], 0, 1, 256)
        plt.plot(np.linspace(0, 1, 256), g_hist_321, 'g')

        # B channel land cover.
        plt.subplot(325)
        b_hist_321 = ndi.histogram(land_covers_images[index][:, :, 2], 0, 1, 256)
        plt.plot(np.linspace(0, 1, 256), b_hist_321, 'b')

        plt.show()


        # RGB channels as bands 312.
        plt.figure(frameon=False)
        plt.suptitle('RGB channels as bands 312. Left: ' + str(cover) + '. Right: Original image.')

        # R channel original image.
        plt.subplot(322)
        plt.plot(np.linspace(0, 1, 256), histograms['3en'], 'r')

        # G channel original image.
        plt.subplot(324)
        plt.plot(np.linspace(0, 1, 256), histograms['1en'], 'g')

        # B channel original image.
        plt.subplot(326)
        plt.plot(np.linspace(0, 1, 256), histograms['2en'], 'b')

        # R channel land cover.
        plt.subplot(321)
        r_hist_754 = ndi.histogram(land_covers_images[index][:, :, 3], 0, 1, 256)
        plt.plot(np.linspace(0, 1, 256), r_hist_754, 'r')

        # G channel land cover.
        plt.subplot(323)
        g_hist_754 = ndi.histogram(land_covers_images[index][:, :, 4], 0, 1, 256)
        plt.plot(np.linspace(0, 1, 256), g_hist_754, 'g')

        # B channel land cover.
        plt.subplot(325)
        b_hist_754 = ndi.histogram(land_covers_images[index][:, :, 5], 0, 1, 256)
        plt.plot(np.linspace(0, 1, 256), b_hist_754, 'b')

        plt.show()

elif show_lc_hist == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Filtering each land cover.
###

# Set up a function for filtering. It got kind of messy. Don't like it.

"""
The way this is going to work, is that we have 5 land covers (water,
city, mountain, crops and vegetation) and we are filtering over the IR
bands (Bands 4, 5 and 7) for everything except city and mountain, visible
will be used to filter city.

By trial and error, what I thought worked best is to filter as follows:
    - Water
    - City (¬Water)
    - Vegetation (¬Water & ¬City)
    - Crops (¬Water & ¬City & ¬Vegetation)

So, for each of the above, auxiliary arrays will be set up for indexing,
which will then be 'combined', and a 'final' array for indexing will be
produced. Then, we will fill an array, noticing that we want, for example,
water to overwrite everything else, so it will be applied last.
"""

# Function: filter.
# Filters the image. Returns a ndarray. Land covers are coloured as follows:
#           Water -> cyan, RGB = (0, 1, 1)
#           City -> yellow, RGB = (1, 1, 0)
#           Vegetation -> green, RGB = (0, 1, 0)
#           Crops -> magenta, RGB = (1, 0, 1)
#           Else -> black, RGB = (0, 0, 0)
# Parameters:
#   bandf - dictionary containing the image bands. Keys must(!) be the band
#           number (string), values must be ndarrays.
def filter(bandf):

    # Water auxiliary arrays.
    index_water_band4 = bandf['4'] > 0.994
    index_water_band5 = bandf['5'] > 0.994
    index_water_band7 = bandf['7'] > 0.994

    # Water index array.
    index_water = np.logical_and(np.logical_and(index_water_band4, index_water_band5), index_water_band7)

    # City auxiliary arrays.
    index_city_band3 = bandf['3'] < 0.319
    index_city_band1 = bandf['1'] < 0.319
    index_city_band2 = bandf['2'] < 0.319

    # City index array.
    index_city = np.logical_and(np.logical_and(index_city_band3, index_city_band1), index_city_band2)

    # Vegetation auxiliary arrays (This land cover had 2 distinct 'colours').
    index_vegetation_band4_1 = np.logical_and(bandf['4'] > 0.009, bandf['4'] < 0.325)
    index_vegetation_band5_1 = np.logical_and(bandf['5'] > 0.341, bandf['5'] < 0.509)
    index_vegetation_band7_1 = np.logical_and(bandf['7'] > 0.513, bandf['7'] < 0.761)

    index_vegetation_band4_2 = np.logical_and(bandf['4'] > 0.351, bandf['4'] < 0.647)
    index_vegetation_band5_2 = np.logical_and(bandf['5'] > 0.565, bandf['5'] < 0.747)
    index_vegetation_band7_2 = np.logical_and(bandf['7'] > 0.746, bandf['7'] < 0.896)

    index_vegetation_aux_1 = np.logical_and(np.logical_and(index_vegetation_band4_1, index_vegetation_band5_1), index_vegetation_band7_1)
    index_vegetation_aux_2 = np.logical_and(np.logical_and(index_vegetation_band4_2, index_vegetation_band5_2), index_vegetation_band7_2)

    # Vegetation index array.
    index_vegetation = np.logical_or(index_vegetation_aux_1, index_vegetation_aux_2)

    # Crops auxiliary arrays (This land cover had 2 distinct 'colours').
    index_crops_band4_1 = np.logical_and(bandf['4'] > 0.451, bandf['4'] < 0.907)
    index_crops_band5_1 = bandf['5'] < 0.099
    index_crops_band7_1 = bandf['7'] < 0.195

    index_crops_band4_2 = np.logical_and(bandf['4'] > 0.451, bandf['4'] < 0.907)
    index_crops_band5_2 = np.logical_and(bandf['5'] > 0.157, bandf['5'] < 0.553)
    index_crops_band7_2 = np.logical_and(bandf['7'] > 0.213, bandf['7'] < 0.579)

    index_crops_aux_1 = np.logical_and(np.logical_and(index_crops_band4_1, index_crops_band5_1), index_crops_band7_1)
    index_crops_aux_2 = np.logical_and(np.logical_and(index_crops_band4_2, index_crops_band5_2), index_crops_band7_2)

    # Crops index array.
    index_crops = np.logical_or(index_crops_aux_1, index_crops_aux_2)

    # Set up lists with index arrays and their corresponding colour.
    indices = [index_crops, index_vegetation, index_city, index_water]
    rgb = [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1]]

    # Create and fill a new array according to indices and colours.
    filtered_image = np.zeros(np.shape(bandf['1']) + (3,))

    for counter, index in enumerate(indices):
        for channel in range(3):
            filtered_image[index, channel] = rgb[counter][channel]

    return filtered_image


dummy_dict = {}

# Ask whether to show the filtered image (cropped) or not.
print(' ')
show_filtered_cropped = input('Show filtered image (cropped)? (y/n): ')

if show_filtered_cropped == str('y'):
    # Store enhanced bands in dummy dictionary.
    for bands in [1, 2, 3, 4, 5, 7]:
        dummy_dict['{}'.format(bands)] = band['{}en'.format(bands)]

    # Apply filter.
    image_shower = filter(dummy_dict)

    plt.figure(frameon=False)
    plt.title('Filtered image (cropped).')
    plt.imshow(image_shower)
    plt.show()
elif show_filtered_cropped == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


'''
Sergio, la siguiente parte deberia funcionar. No se bien que ira a pasar con los
pixeles negros de las bandas originales. Eso me rompia el realce porque al tener
tantos pixeles negros, hay una condicion que no se cumple nunca y te salta IndexError
porque el np.argwhere(cdf < cdf_max) en la linea 231 es vacio, asi que tuve
que agregar un try except para poder hacer el realce. El tema es que cuando quiero
correr la parte de aca abajo se me congela la compu. Por eso dejo dicho aca que,
aunque deberia funcionar todo bien, no pude correrlo y por eso lo dejo comentado.
'''
'''
# Ask whether to show the full filtered image or not.
print(' ')
show_filtered_whole = input('Show filtered image (whole)? (y/n): ')

if show_filtered_whole == str('y'):
    # Enhance whole bands and store them in dummy dictionary.
    for bands in [1, 2, 3, 4, 5, 7]:
        # Create histograms.
        histograms['{}w'.format(bands)] = ndi.histogram(band['{}w'.format(bands)], 0, 1, 256)
        # Enhance.
        band['{}w_en'.format(bands)] = enhancement(band['{}w'.format(bands)], histograms['{}w'.format(bands)])
        # Store.
        dummy_dict['{}'.format(bands)] = band['{}w_en'.format(bands)]

    # Apply filter.
    image_shower = filter(dummy_dict)

    plt.figure(frameon=False)
    plt.title('Filtered image (cropped).')
    plt.imshow(image_shower)
    plt.show()
elif show_filtered_whole == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')
'''

print(' ')
end = input('Press enter to end the script.')
