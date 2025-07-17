# 🧪 H. pylori Detection using Deep Learning

This repository contains a Jupyter notebook for training a deep learning model to detect *Helicobacter pylori* in histopathology images using YOLO-like pipelines.

## YOLO V11

### 📊 Current Results
- **Validation mAP@50**: `0.35`
- Some images in the dataset **may not have bounding boxes** due to the absence of detectable regions.

### 📁 Directory and Dataset Notes
- Ensure that your **validation image path is different from training path** in the `data.yaml` file.
- Folder structure should follow:
  ```
  dataset/
  ├── train/
  │   ├── images/
  │   └── labels/
  └── val/
      ├── images/
      └── labels/
  ```
- Folder structure is automatically created when executing the notebook.

### 📈 Visualizations
- The notebook includes performance plots and prediction visualizations using `matplotlib`.
- Bounding boxes are overlaid on validation images to check prediction accuracy.
- In visualization the image path and label path should be changed and make sure that the image used for visualization is a part of val folder

---

#### 🔧 Notes
- YOLO format annotations are used: `class x_center y_center width height` (all normalized).
- Ensure all dependencies (e.g., TensorFlow, OpenCV, matplotlib) are installed for smooth execution.
