import os
import sys
import math
import numpy as np
from scipy import ndimage as ndi
from scipy import signal as sig
from matplotlib import pyplot as plt
from matplotlib import image as img
from matplotlib import patches as pat

# Turn interactive mode on for pyplot.
plt.ion()


###
# Read bands 1 through 3 from image LandsatTM229_82 and store
# them in numpy arrays. We will display the 'true' image, with
# a rectangle showing the 300 by 300 area selected to use.
# Band 4 will be stored because it's the one will use in this script.
###

'''
The following will be, basically, copy+paste from practico3.py.
'''

# Get current working directory.
default_dir = os.getcwd()

# Set directory to image location, change as necessary.
os.chdir(r"C:\Users\Reinaldo\Desktop\Facu\ProImgSat\Practico 3\LandsatTM229_82")

# Read the header.
with open('header.dat') as file:
    header = file.read()

# Slice the header string to get the desired numbers.
row_string = 'LINES PER IMAGE= '
col_string = 'PIXELS PER LINE= '

header_row_index = header.find(row_string) + len(row_string)
rows = int(header[header_row_index : header_row_index + 4 ]) + 1

header_col_index = header.find(col_string) + len(col_string)
columns = int(header[header_col_index : header_col_index + 4])

# Initialize array to store the image.
image_shower = np.zeros((rows, columns, 3), dtype = np.uint8)

# Fill the array with the corresponding data and store band 4
# in a dictionary with it's corresponding key.
band = {}

for index, bands in enumerate([3, 2, 1, 4]):
    with open('BAND' + str(bands) + '.dat','rb') as band_file:
        band_flat = np.fromfile(band_file, dtype = np.uint8)
        try:
            image_shower[:, :, index] = band_flat.reshape(rows, columns)
        except IndexError:
            band['{}'.format(bands)] = band_flat.reshape(rows, columns)


# Return to default directory. This is to avoid unnecessary conflicts.
os.chdir(default_dir)

print(' ')
# Ask whether to show full image showing area selected.
show_area_selected = input('Display image showing area selected? (y/n): ')

if show_area_selected == str('y'):
    plt.figure(frameon=False)
    plt.title('Area selected shown inside red square.')
    plt.imshow(image_shower)
    plt.gca().add_patch(pat.Rectangle((3450, 2900), 300, 300, linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()

    plt.figure(frameon=False)
    plt.title('Area selected.')
    plt.imshow(image_shower[2900:3200, 3450:3750, :])
    plt.show()

elif show_area_selected == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')

# The whole bands are not needed anymore. Crop and normalize band 4.
band['4'] = np.divide(band['4'][2900:3200, 3450:3750], 255)


###
# Add uniform noise of amplitude A and median 0. A is chosen to be
# 0.1.
###

band['4_noisy'] = band['4'] + np.random.default_rng().uniform(-0.1, 0.1, np.shape(band['4']))

print(' ')
# Ask whether to compare band 4 with and without noise or not
show_noise = input('Compare band 4 with and without noise? (y/n): ')

if show_noise == str('y'):
    plt.figure(frameon=False)

    plt.subplot(121)
    plt.title('Band 4, no noise.')
    plt.imshow(band['4'], 'gray')

    plt.subplot(122)
    plt.title('Band 4, noise.')
    plt.imshow(band['4_noisy'], 'gray')

    plt.show()

elif show_noise == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Set up a filter as a function with an image and kernel as
# arguments. Check that the kernel matrix must not have even dimensions.
###


'''
Function: filter
Filters an image using the kernel given.
Returns filtered image.
Parameters:
    image - image as an ndarray.
    kernel - kernel as an ndarray, odd dimensions (normalized).
'''
def filter(image, kernel):
    # The kernel must be a square matrix.
    if np.shape(kernel)[0] != np.shape(kernel)[1]:
        return None

    # The kernel must have an odd number of rows.
    elif (np.shape(kernel)[0] % 2) == 0:
        return None

    else:
        filtered_image = np.zeros(np.shape(image))
        # Get the index for the element of the kernel that is in the middle.
        middle = int((np.shape(kernel)[0] - 1) / 2)

        for row in range(np.shape(image)[0]):
            for column in range(np.shape(image)[1]):
                try:
                    # Slice the array to get the elements we are interested in.
                    auxiliary_slice = image[row - middle: row + middle + 1, column - middle: column + middle + 1]
                    # Multiply, element-wise and get the sum.
                    filtered_image[row, column] = np.sum(np.multiply(auxiliary_slice, kernel))

                # For edges, we want zeros.
                except IndexError:
                    pass

                except ValueError:
                    pass

        # Kill outliers.
        filtered_image[filtered_image < 0] = 0
        filtered_image[filtered_image > 1] = 1

        return filtered_image


###
# Set up different kernels for smoothing and sharpening.
###


# Smoothing (low pass).
smooth_1 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
smooth_2 = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]]) / 10

'''
Another way to remove noise from an image is to use a median filter. The
way this works is by, instead of using a kernel, using the median of the slice
(see line 150).

Function: median_filter
Filters an image by applying a median filter.
Returns filtered image.
Parameters:
    image - image as an ndarray.
    size - dimension of the square slice to be taken.
'''
def median_filter(image, size):
    if (size % 2) == 0:
        return None

    else:
        middle = int((size - 1) / 2)
        filtered_image = np.zeros(np.shape(image))

        for row in range(np.shape(image)[0]):
            for column in range(np.shape(image)[1]):
                try:
                    # Slice the array to get the elements we are interested in.
                    auxiliary_slice = image[row - middle: row + middle + 1, column - middle: column + middle + 1]
                    # Flatten and sort.
                    auxiliary_slice = np.sort(auxiliary_slice.flatten())
                    # Take the median and replace it in the original image.
                    filtered_image[row, column] = auxiliary_slice[int((size * size - 1) / 2)]
                except IndexError:
                    # For edges, we want zeros.
                    pass

        # Kill outliers.
        filtered_image[filtered_image < 0] = 0
        filtered_image[filtered_image > 1] = 1

        return filtered_image




# Sharpening (high pass).
sharp_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharp_2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

'''
Another way of 'applying' a high pass filter is to first apply a low pass
filter to get an image I_lp. Then take the original image I and, element-wise,
perform the operation 2I - I_lp.

Function: hp_filter
Filters (high pass) an image by applying a low pass filter and then removing
elements corresponding to the low pass filtering.
Returns filtered image.
Parameters:
    image - image as an ndarray.
    kernel - kernel as an ndarray, odd dimensions (normalized).
'''
def hp_filter(image, kernel):
    filtered_image = np.zeros(np.shape(image))
    # Apply low pass filter.
    filtered_image = filter(image, kernel)

    for row in range(np.shape(image)[0]):
        for column in range(np.shape(image)[1]):
            filtered_image[row, column] = 2 * image[row, column] - filtered_image[row, column]

    # Kill outliers.
    filtered_image[filtered_image < 0] = 0
    filtered_image[filtered_image > 1] = 1

    return filtered_image


###
# Apply smoothing to noisy image, and compare to original.
###

print(' ')
# Ask whether to smooth and compare with original or not.
show_smooth = input('Smooth noisy image and compare to original? (y/n): ')

if show_smooth == str('y'):
    # Smoothing.
    band['4_lp1'] = filter(band['4_noisy'], smooth_1)
    band['4_lp2'] = filter(band['4_noisy'], smooth_2)
    band['4_mf'] = median_filter(band['4_noisy'], 3)

    plt.figure(frameon=False)

    plt.subplot(221)
    plt.title('Band 4.')
    plt.imshow(band['4'], 'gray')

    plt.subplot(222)
    plt.title('Smoothing. Kernel 1.')
    plt.imshow(band['4_lp1'], 'gray')

    plt.subplot(223)
    plt.title('Smoothing. Kernel 2.')
    plt.imshow(band['4_lp2'], 'gray')

    plt.subplot(224)
    plt.title('Smoothing. Median filter.')
    plt.imshow(band['4_mf'], 'gray')

    plt.show()

elif show_smooth == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Use scipy.signal.correlate to smooth and compare.
###


print(' ')
# Ask whether to use scipy's correlate and comprate to other smoothings.
show_cor = input('Compare smoothings to scipy\'s correlate? (y/n): ')

if show_cor == str('y'):
    # Smoothing.
    band['4_lp1'] = filter(band['4_noisy'], smooth_1)
    band['4_lp2'] = filter(band['4_noisy'], smooth_2)
    band['4_mf'] = median_filter(band['4_noisy'], 3)
    band['4_cor'] = sig.correlate(band['4_noisy'], smooth_1)

    plt.figure(frameon=False)

    plt.subplot(221)
    plt.title('Correlate. Kernel 1.')
    plt.imshow(band['4_cor'], 'gray')

    plt.subplot(222)
    plt.title('Smoothing. Kernel 1.')
    plt.imshow(band['4_lp1'], 'gray')

    plt.subplot(223)
    plt.title('Smoothing. Kernel 2.')
    plt.imshow(band['4_lp2'], 'gray')

    plt.subplot(224)
    plt.title('Smoothing. Median filter.')
    plt.imshow(band['4_mf'], 'gray')

    plt.show()

elif show_cor == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Apply high pass filter to smoothed image and compare to original.
# Smoothed image is taken to be the one obtained with smooth_1.
###

print(' ')
# Ask whether to sharpen smoothed image and compare to original or not.
show_sharp = input('Sharpen smoothed image (smoothing kernel 1) and compare to original? (y/n): ')

if show_sharp == str('y'):
    # Smoothing.
    band['4_lp1'] = filter(band['4_noisy'], smooth_1)

    # Sharpening.
    band['4_hp1'] = filter(band['4_lp1'], sharp_1)
    band['4_hp2'] = filter(band['4_lp1'], sharp_2)
    band['4_hp'] = hp_filter(band['4_noisy'], smooth_1)

    plt.figure(frameon=False)

    plt.subplot(221)
    plt.title('Band 4.')
    plt.imshow(band['4'], 'gray')

    plt.subplot(222)
    plt.title('Sharpening. Kernel 1.')
    plt.imshow(band['4_hp1'], 'gray')

    plt.subplot(223)
    plt.title('Sharpening. Kernel 2.')
    plt.imshow(band['4_hp2'], 'gray')

    plt.subplot(224)
    plt.title('Sharpening. Smoothing kernel 1.')
    plt.imshow(band['4_hp'], 'gray')

    plt.show()

elif show_sharp == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Same as above, but with a Sobel high pass filter.
###

'''
Function: sobel_filter
Filters (high pass) an image by applying a Sobel filter (edge enhancement).
Returns filtered image.
Parameters:
    image - image as an ndarray.
'''
def sobel_filter(image):
    filtered_image = np.zeros(np.shape(image))

    gradient_1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    for row in range(np.shape(image)[0]):
        for column in range(np.shape(image)[1]):
            try:
                # Slice the array to get the elements we are interested in.
                auxiliary_slice = image[row - 1: row + 2, column - 1: column + 2]
                # Multiply, element-wise and get the sum.
                sum_1 = np.sum(np.multiply(auxiliary_slice, gradient_1))
                sum_2 = np.sum(np.multiply(auxiliary_slice, gradient_2))

                filtered_image[row, column] = math.sqrt(sum_1 * sum_1 + sum_2 * sum_2)

            # For edges, we want zeros.
            except IndexError:
                pass

            except ValueError:
                pass


    # Kill outliers.
    filtered_image[filtered_image < 0] = 0
    filtered_image[filtered_image > 1] = 1

    return filtered_image


print(' ')
# Ask whether to smooth and compare to original or not.
show_sobel = input('Apply Sobel filter to smoothed image and compare to original? (y/n): ')

if show_sobel == str('y'):
    # Smoothing.
    band['4_lp1'] = filter(band['4_noisy'], smooth_1)

    # Sharpening.
    band['4_hp1'] = filter(band['4_lp1'], sharp_1)
    band['4_hp2'] = filter(band['4_lp1'], sharp_2)

    # Apply Sobel filter.
    band['4_sobel'] = sobel_filter(band['4_lp1'])

    plt.figure(frameon=False)

    plt.subplot(221)
    plt.title('Band 4.')
    plt.imshow(band['4'], 'gray')

    plt.subplot(222)
    plt.title('Sharpening. Kernel 1.')
    plt.imshow(band['4_hp1'], 'gray')

    plt.subplot(223)
    plt.title('Sharpening. Kernel 2.')
    plt.imshow(band['4_hp2'], 'gray')

    plt.subplot(224)
    plt.title('Sobel filter.')
    plt.imshow(band['4_sobel'], 'gray')

    plt.show()

elif show_sobel == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Apply high pass filter first, then low pass.
# Kernel sharp_2 is taken.
###


print(' ')
# Ask whether apply high pass filter first or not.
show_invert = input('Apply high pass filter first and compare to original? (y/n): ')

if show_invert == str('y'):
    # Sharpening.
    band['4_hp'] = filter(band['4_noisy'], sharp_2)

    # Smoothing.
    band['4_lp1'] = filter(band['4_hp'], smooth_1)
    band['4_lp2'] = filter(band['4_hp'], smooth_2)

    plt.figure(frameon=False)

    plt.subplot(221)
    plt.title('Band 4.')
    plt.imshow(band['4'], 'gray')

    plt.subplot(222)
    plt.title('Noisy image.')
    plt.imshow(band['4_noisy'], 'gray')

    plt.subplot(223)
    plt.title('Smoothing sharpened image. Kernel 1.')
    plt.imshow(band['4_lp1'], 'gray')

    plt.subplot(224)
    plt.title('Smoothing sharpened image. Kernel 2.')
    plt.imshow(band['4_lp2'], 'gray')

    plt.show()

elif show_invert == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')

print(' ')
end = input('Press enter to end the script.')


