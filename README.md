# Image Editor

This is a configurable image processing application written in Python. It supports a variety of common filters and adjustments, and processes images according to a user-defined JSON configuration file.

## Features

- Apply a sequence of image operations using a config file 
- Works with both RGB and grayscale images
- Manual implementation of filters and adjustments without using high-level image libraries
- Modular and extensible architecture (easy to add new operations)

## Supported Operations

### Filters 

- **Box Blur**: Applies a basic blur with specification on its height and width. (used as "blur" op in the json)
- **Sobel Edge Detection**: Highlights image edges using Sobel X and Y kernels. (used as "sobel" op in the json)
- **Sharpen**: Enhances edges by combining the original image with a blurred version (used as "sharpen" op in the json)

### Adjustments (Per-Pixel)

- **Brightness**: Multiplies each pixel by a given factor to brighten or darken the image. (used as "brightness" op in the json)
- **Contrast**: Increases or decreases contrast relative to the mid-point. (used as "contrat" op in the json)
- **Saturation**: Blends between a grayscale version of the image and the original, allowing color intensity control. (used as "saturation" op in the json)

## Technologies Used

- Python 3.12
- NumPy for numerical operations
- ImageIO for reading and writing images


