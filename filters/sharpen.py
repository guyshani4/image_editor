# import numpy as np
#
# class Sharpen:
#     def __init__(self, alpha):
#         self.alpha = alpha
#         # Standard sharpening kernel (Laplacian-based)
#         self.kernel = np.array([[0, -1, 0],
#                                 [-1, 5, -1],
#                                 [0, -1, 0]], dtype=np.float32)
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
#                     weighted_sum = 0.0
#                     for ky in range(-1, 2):
#                         for kx in range(-1, 2):
#                             ny = y + ky
#                             nx = x + kx
#
#                             if 0 <= ny < height and 0 <= nx < width:
#                                 pixel = image[ny, nx, c]
#                                 weight = self.kernel[ky + 1, kx + 1]
#                                 weighted_sum += pixel * weight
#
#                     # Apply alpha scaling for sharpness intensity
#                     output[y, x, c] = (1 - self.alpha) * image[y, x, c] + self.alpha * weighted_sum
#
#         # Clip values to stay in valid [0, 1] range
#         output = np.clip(output, 0, 1)
#
#         if channels == 1:
#             return output[:, :, 0]
#         return output

# filters/sharpen.py
import numpy as np

class Sharpen:
    """
    Sharpens an image by enhancing the difference between the original image
    and a blurred version of it (Unsharp Masking).

    Parameters:
    -----------
    alpha : float
        Strength of sharpening. Higher values = more sharpness.
    """
    def __init__(self, alpha: float):
        self.alpha = alpha
        # Use a basic 3x3 box blur kernel for smoothing
        self.kernel = np.ones((3, 3), dtype=np.float32) / 9.0

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening to an image using unsharp masking.

        Parameters:
        ------------
        image : np.ndarray
            Input image in range [0, 1] and shape (H, W) or (H, W, 3).

        Returns:
        --------
        np.ndarray
            Sharpened image.
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # grayscale to 3D

        H, W, C = image.shape
        padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
        blurred = np.zeros_like(image)

        # Box blur convolution
        for dy in range(3):
            for dx in range(3):
                blurred += padded[dy:dy+H, dx:dx+W, :]
        blurred /= 9.0

        # Unsharp mask: original + alpha * (original - blurred)
        sharpened = image + self.alpha * (image - blurred)
        sharpened = np.clip(sharpened, 0, 1)

        if sharpened.shape[2] == 1:
            return sharpened[:, :, 0]  # return 2D if grayscale

        return sharpened
