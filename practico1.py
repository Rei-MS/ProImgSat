import os
import sys
import math
import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from matplotlib import image as img


###
# Open image.
###


# Get current working directory.
default_dir = os.getcwd()
# print("Current working dir : %s" % os.getcwd())
# Set directory to image location, change as necessary.
os.chdir(r"C:\Users\Reinaldo\Desktop\Facu\ProImgSat\Practico 1")
# Read the image (MatPlotLib can handle png format only. We use Pillow
# to handle other formats.)
chrome_logo = img.imread('chrome.tif')
# Return to default directory. This is to avoid unnecessary conflicts.
os.chdir(default_dir)


###
# Useful stuff.
###


# Store number of rows, columns and channels.
rows, columns, channels = np.shape(chrome_logo)

# Turn interactive mode on for pyplot.
plt.ion()


###
# Show original image.
###


print(' ')
# Ask whether to show the image or not.
show_original = input('Show original image? (y/n): ')

if show_original == str('y'):
    plt.figure(0, frameon=False)
    plt.title('Original image.')
    plt.imshow(chrome_logo)
    plt.show()
elif show_original == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Get the RGB channels and show them.
###


r_channel = chrome_logo[:, :, 0]
g_channel = chrome_logo[:, :, 1]
b_channel = chrome_logo[:, :, 2]

print(' ')
# Ask whether to show the channels or not.
show_channels = input('Show RGB channels? (y/n): ')

if show_channels == str('y'):
    plt.figure(1, frameon=False, figsize=(12, 6))
    # Subplot for red channel.
    plt.subplot(131)
    plt.title('Red channel.')
    plt.imshow(r_channel, cmap='gist_gray')
    # Subplot for green channel + color bar.
    plt.subplot(132)
    plt.title('Green channel.')
    plt.imshow(g_channel, cmap='gist_gray')
    plt.colorbar(orientation='horizontal')
    # Subplot for blue channel.
    plt.subplot(133)
    plt.title('Blue channel.')
    plt.imshow(b_channel, cmap='gist_gray')

    plt.show()
elif show_channels == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Create and ask whether to display histograms for each channel or not.
###


r_channel_hist = ndi.histogram(r_channel, 0, 255, 255)
g_channel_hist = ndi.histogram(g_channel, 0, 255, 255)
b_channel_hist = ndi.histogram(b_channel, 0, 255, 255)

print(' ')
# Ask whether to show the histograms or not.
show_histogram_channels = input('Show histograms for each channel? (y/n): ')

if show_histogram_channels == str('y'):
    # Red channel histogram plot.
    plt.figure(2, frameon=False, figsize=(10, 5))
    plt.title('Histogram; red channel.')
    plt.plot(np.arange(np.size(r_channel_hist)), r_channel_hist, 'r')
    # Green channel histogram plot.
    plt.figure(3, frameon=False, figsize=(10, 5))
    plt.title('Histogram; green channel.')
    plt.plot(np.arange(np.size(g_channel_hist)), g_channel_hist, 'g')
    # Blue channel histogram plot.
    plt.figure(4, frameon=False, figsize=(10, 5))
    plt.title('Histogram; blue channel.')
    plt.plot(np.arange(np.size(b_channel_hist)), b_channel_hist, 'b')

    plt.show()
elif show_histogram_channels == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Define and apply a linear mapping to RGB channels.
# Also compare before and after mapping histograms for G channel.
###


# Function: map
# Returns a linear map with normalization. (0, 255) -> (0, 1)
# Parameters:
#   channel - Channel one wishes to map as an array.
def map(channel):
    maxdif = float(np.amax(channel) - np.amin(channel))
    mapped_channel = (channel - np.amin(channel)) / maxdif

    return mapped_channel

# Apply map to channels.
e_r_channel = map(r_channel)
e_g_channel = map(g_channel)
e_b_channel = map(b_channel)

# Create histogram for mapped green channel.
e_g_channel_hist = ndi.histogram(e_g_channel, 0, 1, 255)

# Ask whether to compare histograms before and after mapping for the green channel or not.
print(' ')
compare_g_histogram = input('Show before and after map histogram for the G channel? (y/n): ')

if compare_g_histogram == str('y'):
    plt.figure(5, frameon=False, figsize=(10, 6.5))
    # Subplot for histogram before mapping.
    plt.subplot(211)
    plt.title('Histogram; green channel. Before mapping.')
    plt.plot(np.arange(np.size(g_channel_hist)), g_channel_hist, 'g')
    # Subplot for histogram after mapping.
    plt.subplot(212)
    plt.title('Histogram; green channel. After mapping (+normalization).')
    plt.plot(np.arange(np.size(e_g_channel_hist))/255, e_g_channel_hist, 'g')

    plt.show()
elif compare_g_histogram == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Recreate and show image with mapped channels.
###


# Initialize the corresponding array.
e_chrome_logo = np.zeros([rows, columns, 3])

# Assign to it the mapped channels.
e_chrome_logo[:, :, 0] = e_r_channel
e_chrome_logo[:, :, 1] = e_g_channel
e_chrome_logo[:, :, 2] = e_b_channel

# Ask whether to show the mapped image or not.
print(' ')
show_mapped_image = input('Show mapped image? (y/n): ')

if show_mapped_image == str('y'):
    plt.figure(6, frameon=False)
    plt.title('Mapped image.')
    plt.imshow(e_chrome_logo)
    plt.show()
elif show_mapped_image == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')

'''
Respuestas puntos 6 y 7.

Si me fijo, medio que me imagino que el pico para 1.0 deberia ser para las regiones que tienen blanco. Blanco
es RGB = (1, 1, 1).
Supongo que los 2 picos altos que estan a la izquierda del 0.4 deberian ser los que se corresponden a la parte
verde del logo.
Deberia haber algun pico que se corresponda con la parte azul en el centro de la imagen. Tambien deberia haber
algun pico que se corresponda con la parte amarilla del logo.
Si tuviera que adivinar, me deberia guiar por la "cantidad de pixeles."
El pico de 1.0 es el mas alto. Entonces este si deberia ser el de los blancos, porque es la parte que tiene
mayor area en la imagen.
Despues hay 3 que son como medianos (~40000). Estos creo que son de las partes de color mas grandes. Tengo uno
en ~0.12, otro en ~0.35 y otro en ~0.38. Si tuviera que adivinar, diria que el de ~0.38 es para la region verde,
el de ~0.35 es de la amarilla, y el de ~0.12 es de la roja. Si me fijo cuando separe los canales, tenia pixeles
de cada canal en las regiones que corresponden a estos colores, entonces tiene sentido.
Despues tengo uno (~20000) en ~0.24 y otro (~10000) en ~0.49. Dejando los 3 chiquitos que quedan para las
regiones que "combinan" dos colores, estos dos deberian ser para la parte del centro (el circulo azul y el
anillo, que deberia ser blanco pero no lo es.) Si tuviera que adivinar, diria que el de ~0.49 es para el anillo
y el de ~0.24 es para el circulo azul. Esto, porque, el anillo quiere ser blanco pero no lo es, entonces supongo
que no se ve blanco porque el valor no es 1, sino que es ~0.49.
Los 3 chiquitos que quedan (~5000) son en ~0.10, ~0.31 y ~0.34. Supongo que el de ~0.10 es el de la region entre
rojo y verde. El de ~0.34 es el de la region entre amarillo y verde, y el de ~0.31 el de la region entre rojo
y amarillo.

Ahora, si veo en la imagen:
El pico de 1.0 si era para el blanco.
Para los medianos el de ~0.12 si era para el rojo, pero en los otros dos era al reves. ~0.38 es amarillo y ~0.35 es
verde.
Para los otros dos de ~0.24 y de ~0.49 estaba bien, son para el circulo azul y el anillo blanco, respectivamente.
Para los ultimos 3,  el de ~0.10 si era para la region entre verde y rojo. Pero los otros al reves, de nuevo.
El de ~0.34 es para la region entre amarillo y rojo, y el de ~0.31 es para la region entre amarillo y verde.
'''


###
# Create a filter to keep only the green regions.
# In the chrome logo, this corresponds to everything except the red regions.
###


# Function: green_filter
# Returns a new image where the regions for which the green channel has
# a value greater or equal than a certain number are mapped.
# Parameters:
#   image - the image one wishes to filter.
#   value - threshold for the filter.
def green_filter(image, value):
    # Set up a boolean array for indexing.
    index = np.ndarray.__ge__(image[:, :, 1], value)
    # Filter the RGB channels.
    image[index, 0] = 0
    image[index, 1] = 1
    image[index, 2] = 0

    return image

# Apply the filter to the mapped image.
gf_e_chrome_logo = green_filter(np.copy(e_chrome_logo), 0.3)

# Ask whether to show the green filtered image or not.
print(' ')
show_green_filtered_image = input('Show green filtered image? (y/n): ')

if show_green_filtered_image == str('y'):
    plt.figure(7, frameon=False)
    plt.title('Green filtered image.')
    plt.imshow(gf_e_chrome_logo)
    plt.show()
elif show_green_filtered_image == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Filter all three RGB channels.
###


# Function: filter
# Returns a new image. This image will resemble the official chrome logo.
# RGB values for this image were taken from the official icon.svg file.
# The filter will identify the regions and assign RGB values to them.
# Parameters:
#   image - the chrome logo image to filter.
def filter(image):
    # Set up boolean arrays for indexing.
    white_index = image[:, :, 1] > 0.4
    blue_index = image[:, :, 0] < 0.01
    lred_index = np.logical_and(image[:, :, 0] > 0.35, image[:, :, 1] < 0.2)
    dred_index = np.logical_and(image[:, :, 0] < 0.35, image[:, :, 1] < 0.2)
    lgreen_index = np.logical_and(image[:, :, 0] < 0.1, image[:, :, 1] > 0.33)
    dgreen_index = np.logical_and(image[:, :, 0] < 0.1, image[:, :, 2] < 0.075)
    lyellow_index = np.logical_and(image[:, :, 0] > 0.4, np.logical_not(white_index))
    dyellow_index = np.logical_and(image[:, :, 2] > 0.10, image[:, :, 2] < 0.11)

    # Set up arrays with index matrices and rgb values.
    indices = [white_index, blue_index, lred_index, dred_index, lgreen_index,
               dgreen_index, lyellow_index, dyellow_index]
    rgb_raw = [[255, 255, 255], [66, 133, 244], [219, 68, 55],
               [186, 55, 41], [15, 157, 88], [15, 134, 75],
               [255, 205, 64], [231, 168, 49]]
    rgb_values = np.divide(rgb_raw, 255)

    # Filter the RGB channels.
    for counter, index in enumerate(indices):
        for channel in range(3):
            image[index, channel] = rgb_values[counter][channel]

    return image

# Apply filter to the mapped image.
f_e_chrome_logo = filter(np.copy(e_chrome_logo))

# Ask whether to show the filtered image or not.
print(' ')
show_filtered_image = input('Show filtered image? (y/n): ')

if show_filtered_image == str('y'):
    plt.figure(8, frameon=False)
    plt.title('Filtered image.')
    plt.imshow(f_e_chrome_logo)
    plt.show()
elif show_filtered_image == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Highlight the leaves in face.png (Racoon image).
###


# Set directory to image location.
os.chdir(r"C:\Users\Reinaldo\Desktop\Facu\ProImgSat\Practico 1")
# Read the image. Note that it is already normalized.
racoon = img.imread('face.png')
# Return to default directory.
os.chdir(default_dir)


# Ask whether to show the original racoon image or not.
print(' ')
show_racoon = input('Show racoon image? (y/n): ')

if show_racoon == str('y'):
    plt.figure(9, frameon=False)
    plt.title('Racoon (Original).')
    plt.imshow(racoon)
    plt.show()
elif show_racoon == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')

# Ask whether to display histograms for each channel or not.
print(' ')
show_histograms_racoon = input('Show histograms for the racoon image for each channel? (y/n): ')

if show_histograms_racoon == str('y'):
    # Create the histograms
    r_racoon_hist = ndi.histogram(racoon[:, :, 0], 0, 1, 255)
    g_racoon_hist = ndi.histogram(racoon[:, :, 1], 0, 1, 255)
    b_racoon_hist = ndi.histogram(racoon[:, :, 2], 0, 1, 255)

    # Red channel histogram plot.
    plt.figure(10, frameon=False, figsize=(10, 5))
    plt.title('Histogram; red channel.')
    plt.plot(np.divide(np.arange(np.size(r_racoon_hist)), 255), r_racoon_hist, 'r')
    # Green channel histogram plot.
    plt.figure(11, frameon=False, figsize=(10, 5))
    plt.title('Histogram; green channel.')
    plt.plot(np.divide(np.arange(np.size(g_racoon_hist)), 255), g_racoon_hist, 'g')
    # Blue channel histogram plot.
    plt.figure(12, frameon=False, figsize=(10, 5))
    plt.title('Histogram; blue channel.')
    plt.plot(np.divide(np.arange(np.size(b_racoon_hist)), 255), b_racoon_hist, 'b')

    plt.show()
elif show_histograms_racoon == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')

# Function: leaves_highlight
# Returns a new image. This image will have the leaves highlighted.
# Parameters:
#   image - the racoon image.
def leaves_highlight(image):
    # Set up a boolean arrays for indexing.
    green_index = np.logical_and(image[:, :, 1] > image[:, :, 0], image[:, :, 1] > image[:, :, 2])
    dark_index = np.logical_and(image[:, :, 0] < 0.1, image[:, :, 2] < 0.1)
    leaves_index = np.logical_and(green_index, np.logical_not(dark_index))

    # Filter the RGB channels.
    image[leaves_index, 1] = 1

    return image

# Apply the highlight.
highlighted_racoon = leaves_highlight(np.copy(racoon))

# Ask whether to show the highlighted racoon image or not.
print(' ')
show_highlighted_racoon = input('Show highlighted racoon image? (y/n): ')

if show_highlighted_racoon == str('y'):
    plt.figure(13, frameon=False)
    plt.title('Racoon (Leaves highlighted).')
    plt.imshow(highlighted_racoon)
    plt.show()
elif show_highlighted_racoon == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


print(' ')
end = input('Press enter to end the script.')
