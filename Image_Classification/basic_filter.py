from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import os

def apply_blur_filter(image_path, output_path):
    """Applies a Gaussian blur to the image."""
    try:
        img = Image.open(image_path)
        img_resized = img.resize((256, 256))
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=3))
        img_blurred.save(output_path)
        print(f"Blurred image saved as '{output_path}'.")
    except Exception as e:
        print(f"Error processing image: {e}")

def apply_invert_filter(image_path, output_path):
    """Inverts the colors of the image."""
    try:
        img = Image.open(image_path)
        # Convert to RGB to ensure the invert operation works correctly
        img_rgb = img.convert('RGB')
        img_resized = img_rgb.resize((256, 256))
        # Use ImageOps to invert the colors of the image
        img_inverted = ImageOps.invert(img_resized)
        img_inverted.save(output_path)
        print(f"Inverted image saved as '{output_path}'.")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print("Image Processor (type 'exit' to quit)\nAvailable filters: blur, invert")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue

        # This is the crucial part: Choose the filter
        choice = input("Choose filter ('blur' or 'invert'): ").strip().lower()
        base, ext = os.path.splitext(image_path)

        if choice == 'blur':
            output_file = f"{base}_blurred{ext}"
            apply_blur_filter(image_path, output_file)
        elif choice == 'invert':
            output_file = f"{base}_inverted{ext}"
            apply_invert_filter(image_path, output_file)
        else:
            print("Unknown filter choice. Please select 'blur' or 'invert'.")