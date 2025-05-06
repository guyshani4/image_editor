import numpy as np

class Sharpen:
    def __init__(self, alpha):
        self.alpha = alpha
        # Standard sharpening kernel (Laplacian-based)
        self.kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]], dtype=np.float32)

    def apply(self, image):
        height, width = image.shape[0], image.shape[1]

        if image.ndim == 2:
            channels = 1
            image = image[:, :, np.newaxis]
        else:
            channels = image.shape[2]

        output = np.zeros_like(image)

        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    weighted_sum = 0.0
                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            ny = y + ky
                            nx = x + kx

                            if 0 <= ny < height and 0 <= nx < width:
                                pixel = image[ny, nx, c]
                                weight = self.kernel[ky + 1, kx + 1]
                                weighted_sum += pixel * weight

                    # Apply alpha scaling for sharpness intensity
                    output[y, x, c] = (1 - self.alpha) * image[y, x, c] + self.alpha * weighted_sum

        # Clip values to stay in valid [0, 1] range
        output = np.clip(output, 0, 1)

        if channels == 1:
            return output[:, :, 0]
        return output
