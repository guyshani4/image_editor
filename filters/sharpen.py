import numpy as np

from filters.box_blur import BoxBlur


class Sharpen:
    """
    Sharpens an image by enhancing the difference between the original image
    and a blurred version of it (Un-sharp Masking).
    """
    def __init__(self, alpha: float):
        self.alpha = alpha
        # Prompt: For implementing an un-sharp mask in image processing using NumPy,
        # what's a good way to define a blur kernel?
        # Is it better to use np.ones((5,5)) or create a Gaussian kernel manually?
        self.kernel = np.ones((5, 5), dtype=np.float32) / 25.0
        self.blur = BoxBlur(5, 5)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a technique called Un-sharp masking to enhance image sharpness.
        This method sharpens the image by subtracting a blurred version of the image
        from the original. The blur uses a hardcoded 5x5 kernel (corresponding to radius 2) as required.
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # convert black and white (grayscale - 2D) to 3D

        # Box blur convolution
        blurred = self.blur.apply(image)

        # technique that called Un-sharp mask: original + alpha * (original - blurred)
        sharpened = image + self.alpha * (image - blurred)
        sharpened = np.clip(sharpened, 0, 1)

        if sharpened.shape[2] == 1:
            return sharpened[:, :, 0]  # return 2D if input was black and white (grayscale)

        return sharpened
