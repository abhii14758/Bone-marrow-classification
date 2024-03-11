import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Paths to the classification report CSV files
classification_report_paths = [
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Class_Report\DenseNet121.csv',
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Class_Report\Inception.csv',
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Class_Report\MobileNetV2.csv',
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Class_Report\ResNet50.csv',
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Class_Report\ResNet152.csv',
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Class_Report\VGG16.csv',
    r'C:\Users\HP\Desktop\Leukemia Research\Models\Class_Report\Xception.csv',
]

# Model names
model_names = ['DenseNet121', 'InceptionV3', 'MobileNetV2',
               'ResNet50_10_epoch', 'ResNet152', 'VGG_16','Xception']

# Load each classification report into a DataFrame and store them in a dictionary
classification_reports = {}
for path, name in zip(classification_report_paths, model_names):
    classification_reports[name] = pd.read_csv(path)

# Create a font for annotating the images
font = ImageFont.truetype("arial.ttf", 12)

# Initialize variables for image dimensions
max_width = 0
total_height = 0

# Calculate the total height and maximum width of the images
for name, report in classification_reports.items():
    width = len(report.columns) * 100
    height = len(report) * 20
    total_height += height
    if width > max_width:
        max_width = width

# Create a new blank image with the total dimensions
merged_image = Image.new('RGB', (max_width, total_height), color='white')
draw = ImageDraw.Draw(merged_image)

# Paste each classification report into the merged image and annotate with model name
y_offset = 0
for name, report in classification_reports.items():
    width = len(report.columns) * 100
    height = len(report) * 20
    for idx, (_, row) in enumerate(report.iterrows()):
        for jdx, (col_name, value) in enumerate(row.items()):
            x = jdx * 100
            y = y_offset + idx * 20
            draw.text((x, y), f"{col_name}: {value}", fill='black', font=font)
    draw.text((max_width - 100, y_offset), name, fill='black', font=font)
    y_offset += height

# Save the merged image
merged_image.save('merged_classification_reports.png')
