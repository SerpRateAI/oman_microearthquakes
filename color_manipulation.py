# Find the nearest color in a color map
def find_nearest_color(color, color_dict):
    import numpy as np

    # Compute the distance between the color and all the colors in the map
    distances = np.sqrt(np.sum((np.array(list(color_dict.values())) - np.array(color))**2, axis=1))

    # Find the index of the closest color
    index = np.argmin(distances)

    # Return the corresponding scalar value
    return list(color_dict.keys())[index]

# Convert an RGB image to a scalor image
def rgb_to_scalar(image_rgb, color_dict):
    import numpy as np

    ## Initialize the scalar image
    image_sca = np.zeros(image_rgb.shape[:2])

    ## Loop over the pixels
    for i in range(image_rgb.shape[0]):
        for j in range(image_rgb.shape[1]):
            # Find the nearest color
            image_sca[i, j] = find_nearest_color(image_rgb[i, j], color_dict)

    ## Return the scalar image
    return image_sca