import os
import random
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths
csv_file = "Dataset/BBox_List_2017.csv"
images_dir = "Dataset/images"
save_dir = "Dataset/ChestXray8(2)"
os.makedirs(save_dir, exist_ok=True)

# Parameters
target_size = (640, 640)

# Load CSV and clean
df = pd.read_csv(csv_file)
df = df.dropna(subset=["Bbox [x", "y", "w", "h]"])

# Position description helper
def get_position_description(bbox, img_width, img_height):
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    horiz = "left" if center_x < img_width / 3 else "right" if center_x > 2 * img_width / 3 else "central"
    vert = "upper" if center_y < img_height / 3 else "lower" if center_y > 2 * img_height / 3 else "middle"
    return f"{vert} {horiz}"

# Prompt generation
def get_prompt_text_with_location(finding_label, bbox, img_width, img_height):
    if finding_label == "No Finding":
        return "This chest X-ray appears normal."
    else:
        position = get_position_description(bbox, img_width, img_height)
        findings = finding_label.split('|') if '|' in finding_label else finding_label.split(',')
        findings = [f.strip().lower() for f in findings]
        findings_phrase = " and ".join(findings)
        return f"Signs of {findings_phrase} detected in the {position} lung area."

# Resize image and scale bounding box
def resize_image_and_bbox(image, bbox, target_size):
    img_width, img_height = image.size
    scaled_bbox = [
        bbox[0] * target_size[0] / img_width,   # x
        bbox[1] * target_size[1] / img_height,  # y
        bbox[2] * target_size[0] / img_width,   # w
        bbox[3] * target_size[1] / img_height   # h
    ]
    image_resized = image.resize(target_size)
    return image_resized, scaled_bbox

# Main loop
entries = []
anno_id = 0
image_id_map = {}
current_image_id = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(images_dir, row['Image Index'])
    img_path = img_path.replace('Dataset/', '')

    if not os.path.exists(os.path.join('Dataset/', img_path)):
        continue

    finding_label = row['Finding Label']
    bbox = [row['Bbox [x'], row['y'], row['w'], row['h]']]

    # Load image and resize + scale bbox
    img = Image.open(os.path.join('Dataset/', img_path))
    img_resized, scaled_bbox = resize_image_and_bbox(img, bbox, target_size)

    # Assign image ID
    if row['Image Index'] not in image_id_map:
        image_id_map[row['Image Index']] = current_image_id
        current_image_id += 1
    image_id = image_id_map[row['Image Index']]

    # Phrase and prompt
    findings = finding_label.split('|') if '|' in finding_label else finding_label.split(',')
    findings = [f.strip().lower() for f in findings]
    phrase = " and ".join(findings) if finding_label != "No Finding" else "no abnormal findings"
    prompt_text = get_prompt_text_with_location(finding_label, scaled_bbox, target_size[0], target_size[1])

    entry = [
        anno_id,
        image_id,
        0,                        # category_id
        img_path.replace("\\", "/"),
        scaled_bbox,
        phrase,
        prompt_text
    ]
    entries.append(entry)
    anno_id += 1

# Shuffle and split
random.shuffle(entries)
train_entries, temp_entries = train_test_split(entries, test_size=0.3, random_state=42)
val_entries, test_entries = train_test_split(temp_entries, test_size=0.333, random_state=42)

# Save
torch.save(train_entries, os.path.join(save_dir, "ChestXray8_train.pth"))
torch.save(val_entries, os.path.join(save_dir, "ChestXray8_val.pth"))
torch.save(test_entries, os.path.join(save_dir, "ChestXray8_test.pth"))

print(f"Finished! Saved {len(train_entries)} train, {len(val_entries)} val, {len(test_entries)} test entries.")
