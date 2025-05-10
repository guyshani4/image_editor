import numpy as np

class Brightness:
    def __init__(self, value):
        self.value = value  # Brightness scale factor: 1.2 = 20% brighter, 0.8 = decrease

    def apply(self, image):
        """
            Adjusts the brightness of an image.
            Brightness is modified by scaling pixel values:
            - A value > 1.0 makes the image brighter
            - A value < 1.0 makes the image darker
            - A value of 1.0 leaves the image unchanged
        """
        adjusted = image * self.value
        # Prompt: How do I clip a NumPy array to the range [0, 1]
        # to make sure image values stay valid for display or saving?
        adjusted = np.clip(adjusted, 0, 1)
        return adjusted
