import os

dataset_folder = "../datasets_sdb/fernando_ultralytics"
images_folder = os.path.join(os.path.join(dataset_folder, "images"), "train")
labels_folder = os.path.join(os.path.join(dataset_folder, "labels"), "train")
output_dir = "./training_flux/training_images"
selected_label = 5

# PADDING AS FRACTION (e.g., 0.15 adds 15% to each crop edge, if possible)
CROP_PADDING_FRAC = 0.45

if os.path.isdir(images_folder) and os.path.isdir(labels_folder):
    image_files = sorted([
        os.path.join(images_folder, fname)
        for fname in os.listdir(images_folder)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    label_files = sorted([
        os.path.join(labels_folder, fname)
        for fname in os.listdir(labels_folder)
        if fname.lower().endswith(('.txt',))
    ])

    print(f"Found {len(image_files)} images and {len(label_files)} labels.")

    for img_path in image_files:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_folder, f"{img_name}.txt")

        print(img_path, label_path)
        bboxes = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    label = int(parts[0])
                    if label == selected_label:
                        bbox = list(map(float, parts[1:5]))
                        bboxes.append(bbox)
        if len(bboxes) > 0:
            from PIL import Image
            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Open image
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                for i, bbox in enumerate(bboxes):
                    # Each bbox is [x_center, y_center, width, height] in YOLO format (relative)
                    x_center, y_center, width, height = bbox
                    # Convert to absolute pixel coordinates
                    x_center_pix = x_center * img_width
                    y_center_pix = y_center * img_height
                    width_pix = width * img_width
                    height_pix = height * img_height

                    # Add padding in absolute pixel values
                    pad_x = width_pix * CROP_PADDING_FRAC / 2
                    pad_y = height_pix * CROP_PADDING_FRAC / 2

                    left = int(x_center_pix - width_pix / 2 - pad_x)
                    top = int(y_center_pix - height_pix / 2 - pad_y)
                    right = int(x_center_pix + width_pix / 2 + pad_x)
                    bottom = int(y_center_pix + height_pix / 2 + pad_y)

                    # Make sure crop is within image bounds
                    left = max(0, left)
                    top = max(0, top)
                    right = min(img_width, right)
                    bottom = min(img_height, bottom)

                    cropped_img = img.crop((left, top, right, bottom))

                    # Only save the crop if its dimensions are greater than 250x250 px
                    crop_width, crop_height = cropped_img.size
                    if crop_width > 250 and crop_height > 250:
                 
                        crop_filename = f"{img_name}_label{selected_label}_crop{i}.jpg"
                        img.save("./original_images/"+crop_filename)
                        crop_path = os.path.join(output_dir, crop_filename)
                        # Convert RGBA images to RGB before saving as JPEG to avoid OSError
                        cropped_img = cropped_img.convert("RGB")
                        cropped_img.save(os.path.join(output_dir, crop_filename))
                        print(f"Saved crop to: {crop_path}")
                    else:
                        print(f"Skipped crop for {img_name}_label{selected_label}_crop{i} (size {crop_width}x{crop_height}) - too small.")
else:
    print(f"Either images/ or labels/ directory does not exist in {dataset_folder}.")