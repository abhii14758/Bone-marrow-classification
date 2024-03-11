from PIL import Image, ImageDraw, ImageFont

# Load the confusion matrix images and their respective model names
image_paths = [
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Confusion Matrics\DenseNet121.png',
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Confusion Matrics\InceptionV3.png',
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Confusion Matrics\MobileNetV2.png',
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Confusion Matrics\ResNet50_10_epoch.png',
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Confusion Matrics\ResNet152.png',
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Confusion Matrics\VGG_16.png',
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Confusion Matrics\Xception.png',
]

model_names = ['DenseNet121', 'InceptionV3', 'MobileNetV2',
               'ResNet50_10_epoch', 'ResNet152', 'VGG_16', 'Xception']

# Open each image
images = [Image.open(image_path) for image_path in image_paths]

# Get the dimensions of the images
image_widths, image_heights = zip(*(image.size for image in images))

# Calculate the total width and height for the merged image
total_width = sum(image_widths)
max_height = max(image_heights)

# Create a new blank image with the total dimensions
# Increased height for displaying model names and making space for larger font
merged_image = Image.new('RGB', (total_width, max_height + 150))

# Paste each confusion matrix image onto the blank image
draw = ImageDraw.Draw(merged_image)
x_offset = 0
font = ImageFont.truetype("arial.ttf", 120)  # Adjust font size here

for image, model_name, width in zip(images, model_names, image_widths):
    merged_image.paste(image, (x_offset, 0))

    # Calculate the x-coordinate for centering the model name
    text_bbox = draw.textbbox((0, 0), model_name, font=font)
    text_x = x_offset + (width - (text_bbox[2] - text_bbox[0])) // 2

    # Add the model name at the bottom of each image
    draw.text((text_x, max_height), model_name, fill='white', font=font)
    x_offset += width

# Save or display the merged image
merged_image.save('merged_confusion_matrices_with_names_centered.png')
merged_image.show()
