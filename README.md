# 🧪 H. pylori Detection using Deep Learning

This repository contains a Jupyter notebook for training a deep learning model to detect *Helicobacter pylori* in histopathology images using YOLO-like pipelines.

## YOLO V11

### 📊 Current Results
- **Validation mAP@50**: `0.53`
- Some images in the dataset **may not have bounding boxes** due to the absence of detectable regions.
- The model was trained for 220 epochs and is stored in gdrive for validation and training

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
## Super-Resolution Assisted Detection (EDSR)

To improve detection performance on **20× magnification histopathology images**, a super-resolution preprocessing stage using **EDSR (Enhanced Deep Super-Resolution)** is incorporated.

### Motivation
- *H. pylori* regions are small and difficult to detect at lower magnifications.
- Direct detection on 20× images often lacks sufficient spatial detail.
- Super-resolution enhances fine-grained features prior to object detection.

### Methodology
1. **Image Tiling**
   - Each 20× image is sliced into patches of size **640 × 640**
   - This matches the expected YOLO input resolution

2. **Super-Resolution using EDSR**
   - Each patch is passed through an EDSR model with a **2× scaling factor**
   - Output resolution becomes **1280 × 1280**

3. **YOLO Detection**
   - Super-resolved patches are fed into the YOLO detection model
   - Bounding boxes are predicted on the enhanced images

### Benefits
- Enables detection on lower magnification (20×) images
- Reduces dependency on 40× magnification images
- Improves sensitivity to small *H. pylori* regions
- Integrates seamlessly with the existing YOLO pipeline

#### 🔧 Notes
- YOLO format annotations are used: `class x_center y_center width height` (all normalized).
- Ensure all dependencies (e.g., TensorFlow, OpenCV, matplotlib) are installed for smooth execution.
- pip install -r requirements.txt to install all dependencies
- In case gdown does not work use folder of above format
- python -m venv venv
- source venv/bin/activate
- above steps is for creating virtual environment
