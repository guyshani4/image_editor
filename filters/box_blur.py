import numpy as np

class BoxBlur:
    """
    Applies a box blur to an image using NumPy slicing (efficient, non-looping).
    Parameters:
    -----------
    width : int Width of the blur kernel (must be odd)
    height : int Height of the blur kernel (must be odd)
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
           The method performs spatial averaging over a rectangular kernel of size (height x width)
           centered around each pixel. This smooths the image and reduces noise.
           Padding is applied to preserve original image dimensions.
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # convert black and white (grayscale - 2D) to 3D

        H, W, C = image.shape
        pad_h = self.height // 2
        pad_w = self.width // 2

        # make the edges work with kernels - using padding
        # Prompt: Is padding with np.pad using mode='edge' the best practice when applying convolution
        # over images in NumPy? Are there recommended alternatives to handle borders?"
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
        output = np.zeros_like(image)

        # Sum all the shifted versions of the padded updated image
        for dy in range(self.height):
            for dx in range(self.width):
                output += padded[dy:dy+H, dx:dx+W, :]

        # Average
        output /= (self.width * self.height)
        output = np.clip(output, 0, 1)

        if output.shape[2] == 1:
            return output[:, :, 0]  # return 2D if input was black and white (grayscale)
        return output
