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
            image = image[:, :, np.newaxis]  # convert black and white (grayscale - 2D) to 3D

        H, W, C = image.shape
        padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
        blurred = np.zeros_like(image)

        # Box blur convolution
        for dy in range(3):
            for dx in range(3):
                blurred += padded[dy:dy+H, dx:dx+W, :]
        blurred /= 9.0

        # technique that called Un-sharp mask: original + alpha * (original - blurred)
        sharpened = image + self.alpha * (image - blurred)
        sharpened = np.clip(sharpened, 0, 1)

        if sharpened.shape[2] == 1:
            return sharpened[:, :, 0]  # return 2D if input was black and white (grayscale)

        return sharpened
