import numpy as np

class Sharpen:
    """
    Sharpens an image by enhancing the difference between the original image
    and a blurred version of it (Un-sharp Masking).
    """
    def __init__(self, alpha: float):
        self.alpha = alpha
        # Prompt: For implementing an un-sharp mask in image processing using NumPy,
        # what's a good way to define a uniform blur kernel?
        # Is it better to use np.ones((5,5)) or create a Gaussian kernel manually?
        self.kernel = np.ones((5, 5), dtype=np.float32) / 25.0

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a technique called Un-sharp masking to enhance image sharpness.
        This method sharpens the image by subtracting a blurred version of the image
        from the original. The blur uses a hardcoded 5x5 kernel (corresponding to radius 2) as required.
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # convert black and white (grayscale - 2D) to 3D

        H, W, C = image.shape

        # Box blur convolution
        # Prompt: same for BoxBlur padding use.
        padded = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='edge')
        blurred = np.zeros_like(image)

        for dy in range(5):
            for dx in range(5):
                # Prompt: When implementing a box blur using NumPy, is it efficient and accurate to sum over
                # shifted regions using slicing like padded[dy:dy+H, dx:dx+W, :]?
                # Are there more optimized alternatives for applying a uniform blur filter?
                blurred += padded[dy:dy + H, dx:dx + W, :]
        blurred /= 25.0

        # technique that called Un-sharp mask: original + alpha * (original - blurred)
        sharpened = image + self.alpha * (image - blurred)
        sharpened = np.clip(sharpened, 0, 1)

        if sharpened.shape[2] == 1:
            return sharpened[:, :, 0]  # return 2D if input was black and white (grayscale)

        return sharpened
