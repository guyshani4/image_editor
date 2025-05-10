import numpy as np

class Contrast:
    def __init__(self, value: float):
        self.value = value  # Contrast factor: 1.0 = no change, 1.5 = increase, 0.8 = decrease

    def apply(self, image):
        """
            Adjusts the contrast of an image.
            Contrast is modified by scaling pixel values relative to the midpoint (which is 0.5).
            - A value > 1.0 increases contrast
            - A value < 1.0 decreases contrast
            - A value of 1.0 leaves the image unchanged
        """
        midpoint = 0.5
        adjusted = (image - midpoint) * self.value + midpoint
        # Prompt: How do I clip a NumPy array to the range [0, 1]
        # to make sure image values stay valid for display or saving?
        adjusted = np.clip(adjusted, 0, 1)
        return adjusted
