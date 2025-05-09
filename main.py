import argparse
import json
import os
import logging
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import time

from filters.box_blur import BoxBlur
from filters.sobel import Sobel
from filters.sharpen import Sharpen
from adjustments.brightness import Brightness
from adjustments.contrast import Contrast
from adjustments.saturation import Saturation

# Logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(message)s'
)

def prepare_output_path(path, prefix="image", ext="png"):
    """
    Handles output path logic:
    - If it's a directory (or no extension), ensure it exists and generate a unique filename inside.
    - If it's a file and exists, add suffix (_1, _2, ...) until available.
    - Returns a full valid output file path.
    """
    if not path:
        return None

    base, extension = os.path.splitext(path)
    if not extension or path.endswith('/'):
        # Treat as directory
        if not os.path.exists(path):
            create = input(f"❌ Output directory '{path}' does not exist. Create it? [y/n]: ").strip().lower()
            if create == 'y':
                os.makedirs(path)
            else:
                return None
        timestamp = int(time.time())
        return os.path.join(path, f"{prefix}_{timestamp}.{ext}")
    else:
        # Treat as file
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        counter = 1
        new_path = path
        while os.path.exists(new_path):
            new_path = f"{base}_{counter}{extension}"
            counter += 1
        return new_path

def parse_config(path):
    try:
        with open(path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ File '{path}' not found. Please try again.")
        return None
    except json.JSONDecodeError:
        print("❌ Invalid JSON file format.")
        return None

    if 'input' not in config:
        print("❌ Missing 'input' field in config.")
        return None

    if not config.get('output') and not config.get('display', False):
        print("❌ You must provide 'output' or set 'display' to true.")
        return None

    return config

class ImageEditor:
    def __init__(self, config):
        try:
            self.image = iio.imread(config['input']).astype(np.float32) / 255.0
        except Exception as e:
            print(f"❌ Failed to load input image: {e}")
            logging.error(f"Image load failed: {e}")
            raise

        self.operations = config.get('operations', [])
        self.output = config.get('output')
        self.display = config.get('display', False)

    def apply_operations(self):
        for op in self.operations:
            try:
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
                    print(f"⚠️ Unsupported operation: {op_type}")
                    logging.warning(f"Unsupported operation: {op_type}")
            except Exception as e:
                print(f"❌ Failed operation '{op}': {e}")
                logging.error(f"Operation failed: {op}, Error: {e}")

    def run(self):
        self.apply_operations()
        result = (np.clip(self.image, 0, 1) * 255).astype(np.uint8)

        if result.ndim == 3 and result.shape[2] == 4:
            result = result[:, :, :3]

        if self.output:
            self.output = prepare_output_path(self.output)

            try:
                iio.imwrite(self.output, result)
            except Exception as e:
                print(f"❌ Failed to save output image: {e}")
                logging.error(f"Save failed: {e}")

        if self.display:
            plt.imshow(result)
            plt.axis('off')
            plt.show()

if __name__ == '__main__':
    print("📄 Please provide a config JSON file with:")
    print("    • 'input': path to image")
    print("    • 'output': optional path or folder")
    print("    • 'operations': list of operations")
    print("    • 'display': true/false\n")

    while True:
        config_path = input("Enter path to config JSON file: ").strip()
        config = parse_config(config_path)
        if config:
            try:
                editor = ImageEditor(config)
                editor.run()
            except Exception as e:
                print("⚠️ Could not complete image editing. Check logs.")
            break
