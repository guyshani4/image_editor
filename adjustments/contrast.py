import numpy as np


class Contrast:
    def __init__(self, value: float):
        self.value = value  # Contrast factor, e.g., 1.0 = no change, 1.5 = increase, 0.8 = decrease

    """
    Adjusts the contrast of an image.
    Contrast is modified by scaling pixel values relative to the midpoint (0.5).
    - A value > 1.0 increases contrast
    - A value < 1.0 decreases contrast
    - A value of 1.0 leaves the image unchanged
    """
    def apply(self, image):
        midpoint = 0.5
        adjusted = (image - midpoint) * self.value + midpoint
        adjusted = np.clip(adjusted, 0, 1)
        return adjusted
