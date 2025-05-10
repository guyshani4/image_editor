import logging
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

from filters.box_blur import BoxBlur
from filters.sobel import Sobel
from filters.sharpen import Sharpen
from adjustments.brightness import Brightness
from adjustments.contrast import Contrast
from adjustments.saturation import Saturation
import main


class ImageEditor:
    """
    - Loads an image and applies operations specified in a json file.
    - Supports saving and/or displaying the final result.
    """

    SUPPORTED_OPERATIONS = {'blur', 'sobel', 'sharpen', 'brightness', 'contrast', 'saturation'}
    def __init__(self, config):
        """
        - Initializes the editor with config values.
        - Loads the input image and stores output/display preferences.
        """
        try:
            # prompt: How do I read an image in Python and convert it
            # to a normalized NumPy array with float32 values between 0 and 1?
            self.image = iio.imread(config['input']).astype(np.float32) / 255.0
        except Exception as e:
            print(f"❌ Failed to load input image: {e}")
            logging.error(f"Image load failed: {e}")
            raise

        self.operations = config.get('operations', [])
        self.output = config.get('output')
        self.display = config.get('display', False)
        self.validate_operations()

    def apply_operations(self):
        """
        Applies a sequence of image processing operations.
        Each operation modifies the current image. Supported operations:

        - 'blur': Applies box blur with specified 'width' and 'height'.
        - 'sobel': Applies Sobel edge detection.
        - 'sharpen': Sharpens the image using parameter 'alpha'.
        - 'brightness': Adjusts brightness by a 'value'.
        - 'contrast': Adjusts contrast by a 'value'.
        - 'saturation': Adjusts color saturation by a 'value'.

        Unsupported or failed operations are logged and skipped.
        """
        # Prompt: change all those if-else operations to a basic lambda-based OPERATIONS map

        OPERATIONS = {
            'blur': lambda args: BoxBlur(args['width'], args['height']),
            'sobel': lambda args: Sobel(),
            'sharpen': lambda args: Sharpen(args['alpha']),
            'brightness': lambda args: Brightness(args['value']),
            'contrast': lambda args: Contrast(args['value']),
            'saturation': lambda args: Saturation(args['value']),
        }

        for op in self.operations:
            try:
                operation_type = op['type']
                print(f"🔧 Applying {operation_type}...")
                operation = OPERATIONS[operation_type](op)
                self.image = operation.apply(self.image)
            except KeyError as ke:
                # Prompt: Make a pattern to print and logging.error statements consistent, user-friendly,
                # and clearly formatted using emojis like ✅ and ❌.
                # Each message should include the variable context (like operation name or path).
                print(f"❌ Missing parameter for '{operation_type}': {ke}")
                logging.error(f"Missing parameter for '{operation_type}': {ke}")
            except Exception as e:
                print(f"❌ Failed operation '{op}': {e}")
                logging.error(f"Operation failed: {op}, Error: {e}")

    def run(self):
        """
        Executes the complete image editing workflow:

        1. Calls `apply_operations()` to apply all configured image filters and adjustments.
        2. Clips the resulting pixel values to the valid range [0, 1] and converts to uint8 for saving.
        3. Removes alpha channel if present to ensure compatibility with formats like JPEG.
        4. Resolves the output path via `prepare_output_path()` to avoid overwriting files.
        5. Attempts to save the result to disk using imageio. Logs any file I/O errors.
        6. If configured, displays the final image in a matplotlib window.

        This method is typically called once per session.
        """
        self.apply_operations()
        final_image = (np.clip(self.image, 0, 1) * 255).astype(np.uint8)

        if final_image.ndim == 3 and final_image.shape[2] == 4:
            final_image = final_image[:, :, :3]

        if self.output:
            self.output = main.initialize_output_path(self.output)

            try:
                iio.imwrite(self.output, final_image)
            except Exception as e:
                print(f"❌ Failed to save output image: {e}")
                logging.error(f"Save failed: {e}")

        if self.display:
            # Prompt: How do I display an image using matplotlib in Python without axis problems or borders?
            plt.imshow(final_image)
            plt.axis('off')
            plt.show()

    def validate_operations(self):
        """
        Validates all configured operations:
        - Strips whitespace
        - Converts type names to lowercase
        - Checks that operation type is supported
        Removes invalid operations and logs warnings.
        """
        cleaned = []
        for op in self.operations:
            if 'type' not in op:
                logging.warning("Operation missing 'type': %s", op)
                continue
            op_type = op['type'].strip().lower()
            op['type'] = op_type
            if op_type not in self.SUPPORTED_OPERATIONS:
                logging.warning(f"Unsupported operation: {op_type}")
                print(f"⚠️ Skipping unsupported operation: {op_type}")
                continue
            cleaned.append(op)
        self.operations = cleaned