# import numpy as np
#
# class BoxBlur:
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height
#
#     def apply(self, image):
#         img_height, img_width = image.shape[0], image.shape[1]
#
#         if image.ndim == 2:  # grayscale image
#             channels = 1
#             image = image[:, :, np.newaxis]  # convert to (H, W, 1)
#         else:
#             channels = image.shape[2]
#
#         output = np.zeros_like(image)
#
#         for y in range(img_height):
#             for x in range(img_width):
#                 for c in range(channels):
#                     sum_value = 0
#                     count = 0
#
#                     for dy in range(-(self.height // 2), self.height // 2 + 1):
#                         for dx in range(-(self.width // 2), self.width // 2 + 1):
#                             ny = y + dy
#                             nx = x + dx
#
#                             if 0 <= ny < img_height and 0 <= nx < img_width:
#                                 sum_value += image[ny, nx, c]
#                                 count += 1
#
#                     output[y, x, c] = sum_value / count
#
#         if channels == 1:
#             return output[:, :, 0]  # return to (H, W) for grayscale
#         return output

# filters/box_blur.py
import numpy as np

class BoxBlur:
    """
    Applies a box blur to an image using NumPy slicing (efficient, non-looping).

    Parameters:
    -----------
    width : int
        Width of the blur kernel (must be odd)
    height : int
        Height of the blur kernel (must be odd)
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # convert grayscale to 3D

        H, W, C = image.shape
        pad_h = self.height // 2
        pad_w = self.width // 2

        # Pad the image using 'edge' mode
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')

        # Prepare output =  zeroes array
        output = np.zeros_like(image)

        # Sum up all shifted versions of the padded image
        for dy in range(self.height):
            for dx in range(self.width):
                output += padded[dy:dy+H, dx:dx+W, :]

        # Average
        output /= (self.width * self.height)
        output = np.clip(output, 0, 1)

        if output.shape[2] == 1:
            return output[:, :, 0]  # return 2D if input was grayscale
        return output
