import numpy as np

class Brightness:
    def __init__(self, value):
        self.value = value  # Brightness scale factor (e.g. 1.2 = 20% brighter)

    def apply(self, image):
        adjusted = image * self.value  # Scale brightness
        adjusted = np.clip(adjusted, 0, 1)  # Ensure values remain valid
        return adjusted
