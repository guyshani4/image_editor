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
            image = image[:, :, np.newaxis]  # convert black and white (grayscale - 2D) to 3D

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

        # clip to 0-1
        gradient = np.clip(gradient, 0, 1)

        if gradient.shape[2] == 1:
            return gradient[:, :, 0]  # return 2D if input was black and white (grayscale)

        return gradient
