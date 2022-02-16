import os
import sys
import math
import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from matplotlib import image as img
from matplotlib import patches as pat
from itertools import permutations

# Turn interactive mode on for pyplot.
plt.ion()

###
# Read all images .png from GOES, except band number 2.
###


# Get current working directory.
default_dir = os.getcwd()

# Set directory to image location, change as necessary.
os.chdir(r"C:\Users\Reinaldo\Desktop\Google Drive\Imagenes\Goes_png")

# Get the name of every file in this directory.
image_list = os.listdir()

# Store each band in a dictionary.
band = {}

for image_name in image_list:
    try:
        num = image_name[19:21]

        if int(image_name[19]) == 0:
            band['{}'.format(image_name[20])] = img.imread(image_name)
        else:
            band['{}'.format(image_name[19:21])] = img.imread(image_name)

    except IndexError:
        continue

# Kill band number 2.
del band['2']

# Return to default directory. This is to avoid unnecessary conflicts.
os.chdir(default_dir)


###
# Enhance each band and show them.
###

# Function - enhancement
# Returns a band mapped via a linear enhancement (percentile).
# Parameters:
#   v_band - the band one wishes to map.
#   percentage - percentage.
def enhancement(v_band, percentage = 2):
    # Create histogram of the band.
    histogram = ndi.histogram(v_band, 0, 1, 256)

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

# Enhance bands (2% linear map).
for key, value in band.items():
    band['{}'.format(key)] = enhancement(value)


print(' ')
# Ask whether to show each band or not.
show_e_band = input('Show enhanced bands? (y/n): ')

if show_e_band == str('y'):
    for key, value in band.items():
        f = plt.figure()
        f.suptitle('Band number ' + key + ', after 2' + r"%" + ' linear mapping.')

        f.add_subplot(1,2, 1)
        plt.title('Band number ' + key)
        plt.imshow(value, cmap='gist_gray')

        f.add_subplot(1,2, 2)
        plt.title('Band number ' + key + ' inverted')
        plt.imshow(1-value, cmap='gist_gray')

        plt.show()

        print(' ')
        hold = input('Press enter to continue.')

elif show_e_band == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Keep bands corresponding to South America.
###

# Kill bands not corresponding to South America. These correspond to Central America.
for key in [1, 3, 5]:
    del band['{}'.format(key)]

'''
Veo que las nubes se ven claras, salvo por las bandas 4 y 6.
'''


# Invert bands 4 & 6 so that clouds are white.
for key in [4, 6]:
    band['{}'.format(key)] = 1 - band['{}'.format(key)]


###
# Set up different RGB combinations as to highligh different variables.
###

'''
First, water vapor.
We can see that the bands that relate to water vapor, along with their sample use are:
    Band 8 - High-level atmospheric water vapor, winds, rainfall.
    Band 9 - Mid-level atmospheric water vapor, winds, rainfall.
    Band 10 - Low-level atmospheric water vapor, winds, rainfall.
So, those bands will be used for RGB combinations.
'''

print(' ')
# Ask whether to show water vapor RGB combinations or not.
show_watervapor = input('Show water vapor RGB combinations? (y/n): ')

if show_watervapor == str('y'):
    image_shower = np.zeros([np.shape(band['4'])[0], np.shape(band['4'])[1], 3])
    perm = list(permutations(range(3)))
    rgb_link = ['R', 'G', 'B']
    for index in range(6):
        image_shower[:, :, perm[index][0]] = band['8']
        image_shower[:, :, perm[index][1]] = band['9']
        image_shower[:, :, perm[index][2]] = band['10']

        plt.figure(frameon=False)
        plt.title('Water Vapor. Band 8 as ' + rgb_link[perm[index][0]] + ', band 9 as ' + rgb_link[perm[index][1]] + ', band 10 as ' + rgb_link[perm[index][2]])
        plt.imshow(image_shower)
        plt.show()

        print(' ')
        hold = input('Press enter to continue.')

elif show_watervapor == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


'''
Second, snow and cirrus clouds.
We can see that the bands that relate to snow and cirrus clouds, along with their sample use are:
    Band 4 - Daytime cirrus cloud.
    Band 6 - Daytime land/cloud properties, particle size, vegetation, snow.
    Band 7 - Surface and cloud, fog at night, fire, winds.
So, those bands will be used for RGB combinations.
'''

print(' ')
# Ask whether to show snow and cirrus clouds RGB combinations or not.
show_snow = input('Show snow and cirrus clouds RGB combinations? (y/n): ')

if show_snow == str('y'):
    image_shower = np.zeros([np.shape(band['4'])[0], np.shape(band['4'])[1], 3])
    perm = list(permutations(range(3)))
    rgb_link = ['R', 'G', 'B']
    for index in range(6):
        image_shower[:, :, perm[index][0]] = band['4']
        image_shower[:, :, perm[index][1]] = band['6']
        image_shower[:, :, perm[index][2]] = band['7']

        plt.figure(frameon=False)
        plt.title('Snow and Cirrus clouds. Band 4 as ' + rgb_link[perm[index][0]] + ', band 6 as ' + rgb_link[perm[index][1]] + ', band 7 as ' + rgb_link[perm[index][2]])
        plt.imshow(image_shower)
        plt.show()

        print(' ')
        hold = input('Press enter to continue.')

elif show_snow == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')

'''
Third, convective clouds, stratus clouds and surface.
We can see that the bands that relate to convective clouds, stratus clouds and surface, along with their sample use are:
    Band 6 - Daytime land/cloud properties, particle size, vegetation, snow.
    Band 7 - Surface and cloud, fog at night, fire, winds.
    Band 13 - Surface and cloud.
So, those bands will be used for RGB combinations.
'''

print(' ')
# Ask whether to show convective clouds, stratus clouds and surface RGB combinations or not.
show_cscloud = input('Show convective clouds, stratus clouds and surface RGB combinations? (y/n): ')

if show_cscloud == str('y'):
    image_shower = np.zeros([np.shape(band['4'])[0], np.shape(band['4'])[1], 3])
    perm = list(permutations(range(3)))
    rgb_link = ['R', 'G', 'B']
    for index in range(6):
        image_shower[:, :, perm[index][0]] = band['6']
        image_shower[:, :, perm[index][1]] = band['7']
        image_shower[:, :, perm[index][2]] = band['13']

        plt.figure(frameon=False)
        plt.title('Convective & stratus clouds, surface. Band 6 as ' + rgb_link[perm[index][0]] + ', band 7 as ' + rgb_link[perm[index][1]] + ', band 13 as ' + rgb_link[perm[index][2]])
        plt.imshow(image_shower)
        plt.show()

        print(' ')
        hold = input('Press enter to continue.')

elif show_cscloud == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


'''
Sergio, en lo que sigue no pude distinguir ciudades solamente por la imagen.
El salar al principio pense que era un lago.
'''

print(' ')
# Ask whether to show surface elements (lakes, rivers, cities, salt flats, mountains) or not.
show_surfel = input('Show surface elements? (y/n): ')

if show_surfel == str('y'):
    image_shower = np.zeros([np.shape(band['4'])[0], np.shape(band['4'])[1], 3])
    image_shower[:, :, 0] = band['7']
    image_shower[:, :, 1] = band['6']
    image_shower[:, :, 2] = band['13']

    # Cut to keep desired area.
    image_shower = image_shower[555:, 425:, :]

    # Set up arrays for polygons.
    mountain = np.array([[63,385], [55,346], [28,337], [33,291], [104,143], [155,176]])
    river = np.array([[323,356], [377,281], [436,233], [466,152], [561,164], [565,48], [524,35], [417,165], [354,165], [270,300]])
    salt = np.array([[122,67], [112,54], [91,58], [88,109], [143,109], [142,79]])
    lake1 = np.array([[122,67], [146,67], [146,30], [122,30]])
    lake2 = np.array([[393,393], [486,313], [460,303], [384,382]])

    plt.figure(frameon=False)
    plt.title('Surface elements. Band 7 as R, band 6 as G, band 13 as B.')
    plt.imshow(image_shower)
    plt.gca().add_patch(pat.Polygon(mountain, closed=True, linewidth=1, edgecolor='r', facecolor='none', label='Mountains.'))
    plt.gca().add_patch(pat.Polygon(river, closed=True, linewidth=1, edgecolor='b', facecolor='none', label='Rivers.'))
    plt.gca().add_patch(pat.Polygon(salt, closed=True, linewidth=1, edgecolor='k', facecolor='none', label='Salt flats.'))
    plt.gca().add_patch(pat.Polygon(lake1, closed=True, linewidth=1, edgecolor='c', facecolor='none', label='Lakes.'))
    plt.gca().add_patch(pat.Polygon(lake2, closed=True, linewidth=1, edgecolor='c', facecolor='none'))
    plt.legend(loc='upper center')
    plt.show()

elif show_surfel == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


print(' ')
end = input('Press enter to end the script.')
