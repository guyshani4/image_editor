import numpy as np

class Brightness:
    def __init__(self, value):
        self.value = value  # Brightness scale factor (e.g. 1.2 = 20% brighter)

    """
    Adjusts the brightness of an image.
    Brightness is modified by scaling pixel values:
    - A value > 1.0 makes the image brighter
    - A value < 1.0 makes the image darker
    - A value of 1.0 leaves the image unchanged
    """
    def apply(self, image):
        adjusted = image * self.value  # Scale brightness
        adjusted = np.clip(adjusted, 0, 1)  # Ensure values remain valid
        return adjusted
