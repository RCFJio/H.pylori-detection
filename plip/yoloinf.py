import cv2
import os

def draw_yolo_annotations(image_path, label_path, output_path):
    # 1. Load the image to get dimensions
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 2. Read the YOLO label file
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # YOLO format: class_id x_center y_center width height
        parts = line.strip().split()
        class_id = parts[0]
        x_c, y_c, w, h = map(float, parts[1:])

        # 3. Convert normalized coordinates back to pixel coordinates
        # Center X, Center Y, Width, Height -> x_min, y_min, x_max, y_max
        canvas_width = x_c * width
        canvas_height = y_c * height
        box_width = w * width
        box_height = h * height

        x_min = int(canvas_width - (box_width / 2))
        y_min = int(canvas_height - (box_height / 2))
        x_max = int(canvas_width + (box_width / 2))
        y_max = int(canvas_height + (box_height / 2))

        # 4. Draw the bounding box
        color = (0, 255, 0)  # Green box
        thickness = 2
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

        # Optional: Add class ID text
        cv2.putText(image, f"Class: {class_id}", (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 5. Save or Show the result
    cv2.imwrite(output_path, image)
    print(f"Saved visualized image to: {output_path}")

# Example Usage:
img_path = r"D:\MTech Project\code\model\test\images\1963bbe3-0008.jpg"
label_path=r"D:\MTech Project\code\model\test\labels\1963bbe3-0008.txt"
draw_yolo_annotations(img_path, label_path, 'annotated_image_2.jpg')