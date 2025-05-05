import numpy as np

class BoxBlur:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def apply(self, image):
        height, width = image.shape[0], image.shape[1]

        if image.ndim == 2:  # grayscale image
            channels = 1
            image = image[:, :, np.newaxis]  # convert to (H, W, 1)
        else:
            channels = image.shape[2]

        output = np.zeros_like(image)

        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    sum_value = 0
                    count = 0

                    for dy in range(-(self.height // 2), self.height // 2 + 1):
                        for dx in range(-(self.width // 2), self.width // 2 + 1):
                            ny = y + dy
                            nx = x + dx

                            if 0 <= ny < height and 0 <= nx < width:
                                sum_value += image[ny, nx, c]
                                count += 1

                    output[y, x, c] = sum_value / count

        if channels == 1:
            return output[:, :, 0]  # return to (H, W) for grayscale
        return output
