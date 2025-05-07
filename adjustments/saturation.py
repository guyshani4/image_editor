import numpy as np


class Saturation:
    def __init__(self, value: float):
        self.value = value

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
             Adjusts the saturation of an RGB image.

            Saturation is modified by blending each pixel with its grayscale version:
            - A value > 1.0 increases saturation (more vivid colors)
            - A value < 1.0 decreases saturation (more washed out)
            - A value of 0.0 turns the image fully grayscale
        """
        if image.ndim != 3 or image.shape[2] != 3:
            # For grayscale, saturation does nothing
            return image

        # Convert to grayscale using luminosity method
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis]

        # Interpolate between grayscale and original image
        adjusted = gray * (1 - self.value) + image * self.value
        adjusted = np.clip(adjusted, 0, 1)
        return adjusted
