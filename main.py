import json
import os
import logging
import time
import imageEditor

# Logging setup
# Prompt: How do I set up a logging system in Python that logs errors to a file,
# and ensure the log directory is created if it doesn't exist?
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(message)s'
)

def initialize_output_path(path, prefix="image", ext="png"):
    """
    Handles output path:
    - If it's a directory (or no extension), ensure it exists and generate a unique filename inside.
    - If it's a file and exists, add suffix (_1, _2, ...) until available.
    - Returns a full valid output file path.
    """
    if not path:
        return None

    base, extension = os.path.splitext(path)
    if not extension or path.endswith('/'):
        # Treat as directory
        # How do I generate a unique filename with a timestamp and join it to a directory path in Python?
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
    """
    - Loads and validates a JSON configuration file.
    - Ensures required fields ('input', and 'output' or 'display') exist.
    - Returns the config dictionary if valid, otherwise None.
    """
    try:
        with open(path, 'r') as f:
            config = json.load(f)
    # Prompt: are there any more relevant errors can occur when trying
    # to get a "config.json" file in a path from the user?
    except FileNotFoundError:
        # Prompt: same as in ImageEditor class.
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
            # Ask user whether to display or save result
            choice = input("Do you want to display the result or save it to a file? (display/save): ").strip().lower()
            if choice == 'display':
                config['display'] = True
                config['output'] = None
            elif choice == 'save':
                config['display'] = False
            else:
                print("❌ Invalid choice. Please type 'display' or 'save'.")
                continue
        if config:
            try:
                editor = imageEditor.ImageEditor(config)
                editor.run()
            except Exception as e:
                print("⚠️ Could not complete image editing. Check logs.")
            break
