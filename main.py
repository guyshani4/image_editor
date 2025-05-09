import argparse
import json
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

from filters.box_blur import BoxBlur
from filters.sobel import Sobel
from filters.sharpen import Sharpen
from adjustments.brightness import Brightness
from adjustments.contrast import Contrast
from adjustments.saturation import Saturation


def parse_config(path):
    """
    Parses the JSON configuration file.
    """
    with open(path, 'r') as f:
        config = json.load(f)

    # Validation
    if 'input' not in config:
        raise ValueError("Missing 'input' field in config.")
    # Not sure if to delete it
    if not config.get('output') and not config.get('display', False):
        raise ValueError("You must provide either an 'output' path or set 'display' to true.")

    return config


class ImageEditor:
    """
        Loads an image and applies a series of operations specified in a config.
    """
    def __init__(self, config):
        self.image = iio.imread(config['input']).astype(np.float32) / 255.0
        self.operations = config.get('operations', [])
        self.output = config.get('output')
        self.display = config.get('display', False)

    def apply_operations(self):
        """
            Applies all operations in sequence from the config.
        """
        for op in self.operations:
            op_type = op['type']
            print(f"🔧 Applying {op_type}...")

            if op_type == 'blur':
                self.image = BoxBlur(op['width'], op['height']).apply(self.image)
            elif op_type == 'sobel':
                self.image = Sobel().apply(self.image)
            elif op_type == 'sharpen':
                self.image = Sharpen(op['alpha']).apply(self.image)
            elif op_type == 'brightness':
                self.image = Brightness(op['value']).apply(self.image)
            elif op_type == 'contrast':
                self.image = Contrast(op['value']).apply(self.image)
            elif op_type == 'saturation':
                self.image = Saturation(op['value']).apply(self.image)
            else:
                raise ValueError(f"Unsupported operation: {op_type}")

    def run(self):
        self.apply_operations()
        result = (np.clip(self.image, 0, 1) * 255).astype(np.uint8)

        if result.ndim == 3 and result.shape[2] == 4:
            result = result[:, :, :3]

        if self.output:
            iio.imwrite(self.output, result)

        if self.display:
            plt.imshow(result)
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    config_path = input("Enter path to config JSON file: ").strip()

    try:
        config = parse_config(config_path)
        editor = ImageEditor(config)
        editor.run()
    except Exception as e:
        print(f"❌ Error: {e}")

