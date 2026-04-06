import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QSlider, QSizePolicy)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Import your pipeline
from pipeline import run_inference_pipeline

# ==========================================
# 1. THE BACKGROUND WORKER
# ==========================================
class InferenceWorker(QThread):
    finished_signal = pyqtSignal(list)  # We only need it to hand back the coordinates now!
    error_signal = pyqtSignal(str)

    def __init__(self, input_path, model_path, output_path):
        super().__init__()
        self.input_path = input_path
        self.model_path = model_path
        self.output_path = output_path

    def run(self):
        try:
            # Run the heavy math on the DGX
            coords = run_inference_pipeline(self.input_path, self.model_path, self.output_path)
            self.finished_signal.emit(coords)
        except Exception as e:
            self.error_signal.emit(str(e))

# ==========================================
# 2. THE MAIN DESKTOP WINDOW
# ==========================================
class HPyloriDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("H. pylori Detector - Dynamic WSI Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        self.model_weights = "./saved_models/hpylori_best_model.safetensors"
        self.current_image_path = None
        
        # New State Variables to hold the AI's memory
        self.raw_predictions = [] 
        self.base_pixmap = None
        self.annotated_pixmap = None
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Top UI: Buttons and Slider ---
        button_layout = QHBoxLayout()
        
        self.upload_btn = QPushButton("📂 Load Tissue Slide")
        self.upload_btn.setFixedSize(180, 40)
        self.upload_btn.setStyleSheet("font-weight: bold;")
        self.upload_btn.clicked.connect(self.load_image)
        
        self.analyze_btn = QPushButton("🔍 Run AI Analysis")
        self.analyze_btn.setFixedSize(180, 40)
        self.analyze_btn.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold;")
        self.analyze_btn.clicked.connect(self.start_inference)
        self.analyze_btn.setEnabled(False)
        
        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        
        # --- NEW: The Confidence Slider ---
        self.slider_layout = QVBoxLayout()
        self.slider_label = QLabel("Confidence Threshold: 30%")
        self.slider_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(30, 100) # Starts at 35 because your NMS threshold drops anything lower
        self.conf_slider.setValue(30)
        self.conf_slider.setEnabled(False)
        
        # Crucial UX trick: Only redraw the image when the user lets go of the mouse!
        # This prevents the UI from lagging when dragging the slider over massive images.
        self.conf_slider.setTracking(False) 
        self.conf_slider.valueChanged.connect(self.update_annotations)
        
        self.slider_layout.addWidget(self.slider_label)
        self.slider_layout.addWidget(self.conf_slider)
        
        # Add everything to the top bar
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.analyze_btn)
        button_layout.addSpacing(20)
        button_layout.addLayout(self.slider_layout)
        button_layout.addStretch()
        button_layout.addWidget(self.status_label)

        # --- Bottom UI: The WSI Viewer ---
        self.image_viewer = QLabel("No Image Selected")
        self.image_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_viewer.setStyleSheet("background-color: #2b2b2b; border: 2px solid #555;")
        
        # ---> THE 3-LINE LAYOUT FIX <---
        self.image_viewer.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_viewer.setMinimumSize(1, 1) 
        # -------------------------------
        
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_viewer, stretch=1)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Tissue Slide", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if file_name:
            self.current_image_path = file_name
            # Load the pristine image into memory
            self.base_pixmap = QPixmap(self.current_image_path)
            self.display_pixmap(self.base_pixmap)
            
            self.status_label.setText(f"Loaded: {os.path.basename(file_name)}")
            self.analyze_btn.setEnabled(True)
            self.conf_slider.setEnabled(False)
            self.raw_predictions = []

    def start_inference(self):
        if not os.path.exists(self.model_weights):
            QMessageBox.critical(self, "Error", f"Model weights not found at:\n{self.model_weights}")
            return

        self.upload_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.conf_slider.setEnabled(False)
        self.analyze_btn.setText("⏳ Processing...")
        self.status_label.setText("AI is scanning the slide. Please wait...")

        # We pass a dummy output path since the GUI is doing the drawing now
        self.worker = InferenceWorker(self.current_image_path, self.model_weights, "./temp_discard.jpg")
        self.worker.finished_signal.connect(self.on_inference_success)
        self.worker.error_signal.connect(self.on_inference_error)
        self.worker.start()

    def on_inference_success(self, coordinates):
        """Saves the raw AI memory and triggers the first draw."""
        self.raw_predictions = coordinates
        
        # Unlock the UI
        self.upload_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🔍 Run AI Analysis")
        self.conf_slider.setEnabled(True)
        
        # Draw the circles for the first time!
        self.update_annotations()

    def update_annotations(self):
        """Dynamically redraws the cyan targets based on the slider value."""
        if not self.raw_predictions or self.base_pixmap is None:
            return

        threshold = self.conf_slider.value() / 100.0
        self.slider_label.setText(f"Confidence Threshold: {int(threshold * 100)}%")

        valid_pts = [item for item in self.raw_predictions if item['conf'] >= threshold]

        temp_pixmap = self.base_pixmap.copy()
        
        painter = QPainter(temp_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # --- NEW: DYNAMIC SCALING ---
        # Look at how wide the raw image is and scale the pen accordingly
        img_width = temp_pixmap.width()
        # ---> CLEAN DOT-ONLY DRAWING <---
        """scale = max(1, temp_pixmap.width() // 800)
        
        # We only need the bright green brush now!
        green_brush = QColor(0, 255, 0)
        
        for item in valid_pts:
            pt = item['coord']
            gx, gy = int(pt[0]), int(pt[1])
            
            # Draw ONLY the dynamically scaled center dot
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(green_brush)
            
            # scale*2 and scale*4 keeps the dot visible but not overwhelming
            painter.drawEllipse(gx - scale*2, gy - scale*2, scale * 4, scale * 4)
            
        painter.end()"""
        # ---> CIRCLE + DOT DRAWING <---
        scale = max(1, temp_pixmap.width() // 800)
        
        # Define the sizes
        radius = 10 * scale      # Size of the outer ring
        thickness = 3 * scale    # Thickness of the line
        
        # Define the colors
        cyan_pen = QPen(QColor(0, 255, 255), thickness)
        green_brush = QColor(0, 255, 0)
        
        for item in valid_pts:
            pt = item['coord']
            gx, gy = int(pt[0]), int(pt[1])
            
            # 1. Draw the cyan outline first
            painter.setPen(cyan_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush) # Ensures the circle is hollow
            painter.drawEllipse(gx - radius, gy - radius, radius * 2, radius * 2)
            
            # 2. Draw the solid green dot in the center
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(green_brush)
            painter.drawEllipse(gx - scale*2, gy - scale*2, scale * 4, scale * 4)
            
        painter.end()

        self.annotated_pixmap = temp_pixmap
        self.display_pixmap(self.annotated_pixmap)
        
        self.status_label.setText(f"✅ Displaying {len(valid_pts)} bacteria (≥ {int(threshold*100)}% Conf)")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #0078D7;")

    def on_inference_error(self, error_message):
        QMessageBox.critical(self, "AI Pipeline Error", str(error_message))
        self.display_pixmap(self.base_pixmap)
        self.status_label.setText("❌ Analysis Failed.")
        self.upload_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🔍 Run AI Analysis")

    def display_pixmap(self, pixmap):
        """Scales the current pixmap to perfectly fit the window."""
        if pixmap is None: return
        scaled_pixmap = pixmap.scaled(
            self.image_viewer.width(), 
            self.image_viewer.height(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_viewer.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Redraws the scaled image if the user resizes the desktop window."""
        if self.annotated_pixmap:
            self.display_pixmap(self.annotated_pixmap)
        elif self.base_pixmap:
            self.display_pixmap(self.base_pixmap)
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = HPyloriDetectorApp()
    window.show()
    sys.exit(app.exec())