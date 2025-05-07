# import numpy as np
#
# class Sobel:
#     def __init__(self):
#         self.kernel_x = np.array([[-1, 0, 1],
#                                   [-2, 0, 2],
#                                   [-1, 0, 1]], dtype=np.float32)
#
#         self.kernel_y = np.array([[1, 2, 1],
#                                   [0, 0, 0],
#                                   [-1, -2, -1]], dtype=np.float32)
#
#     def apply(self, image):
#         height, width = image.shape[0], image.shape[1]
#
#         if image.ndim == 2:
#             channels = 1
#             image = image[:, :, np.newaxis]
#         else:
#             channels = image.shape[2]
#
#         output = np.zeros_like(image)
#
#         for y in range(height):
#             for x in range(width):
#                 for c in range(channels):
#                     gx = 0.0
#                     gy = 0.0
#
#                     for ky in range(-1, 2):
#                         for kx in range(-1, 2):
#                             ny = y + ky
#                             nx = x + kx
#
#                             if 0 <= ny < height and 0 <= nx < width:
#                                 pixel = image[ny, nx, c]
#                                 gx += pixel * self.kernel_x[ky + 1, kx + 1]
#                                 gy += pixel * self.kernel_y[ky + 1, kx + 1]
#
#                     edge_strength = np.sqrt(gx ** 2 + gy ** 2)
#                     output[y, x, c] = edge_strength
#
#         output = output / output.max()
#
#         if channels == 1:
#             return output[:, :, 0]
#         return output

# filters/sobel.py
import numpy as np

class Sobel:
    """
    Applies the Sobel edge detection filter to an image.
    It highlights areas of high spatial gradient (edges).
    """
    def __init__(self):
        # Standard Sobel kernels for horizontal and vertical edges
        self.kernel_x = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=np.float32)

        self.kernel_y = np.array([[-1, -2, -1],
                                  [ 0,  0,  0],
                                  [ 1,  2,  1]], dtype=np.float32)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # convert grayscale to 3D

        H, W, C = image.shape
        padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
        gradient = np.zeros_like(image)

        # Apply Sobel filter per channel
        for c in range(image.shape[2]):
            gx = np.zeros((H, W), dtype=np.float32)
            gy = np.zeros((H, W), dtype=np.float32)

            for i in range(3):
                for j in range(3):
                    weight_x = self.kernel_x[i, j]
                    weight_y = self.kernel_y[i, j]
                    region = padded[i:i+H, j:j+W, c]
                    gx += region * weight_x
                    gy += region * weight_y

            # Combine horizontal and vertical gradients
            gradient[:, :, c] = np.sqrt(gx**2 + gy**2)

        # Normalize and clip
        gradient = np.clip(gradient, 0, 1)

        if gradient.shape[2] == 1:
            return gradient[:, :, 0]  # return 2D if grayscale

        return gradient
