from PyQt6.QtWidgets import QMainWindow
from PyQt6 import uic
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from pubsub import pub
import logging
import cv2


from message_types import Topics

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # "w" overwrites the file; use "a" to append
)

class MainWindowUI(QMainWindow):
    def __init__(self):
        super(MainWindowUI, self).__init__()
        self.ui = uic.loadUi("main.ui", self)
        self.ui.showMaximized()
        self.ui.setWindowTitle("Image Descriptor")
        
        # Image storage
        self.corner_image = None
        self.matching_image1 = None
        self.matching_image2 = None
        self.sift_image = None
        
        # Setup button connections
        self.setup_connections()
        
        # Subscribe to PubSub events
        self.setup_subscriptions()
    
    def setup_connections(self):
        # Corner Detection tab connections
        self.ui.LoadCornerBtn.clicked.connect(self.load_corner_image)
        self.ui.ApplyCornerBtn.clicked.connect(self.apply_corner_detection)
        
        # Matching tab connections
        self.ui.MatchingLoad1Btn.clicked.connect(self.load_matching_image1)
        self.ui.MatchingLoad2Btn.clicked.connect(self.load_matching_image2)
        self.ui.MatchingApplyBtn.clicked.connect(self.apply_matching)
        
        # SIFT tab connections
        self.ui.SiftLoadBtn.clicked.connect(self.load_sift_image)
        self.ui.SiftApplyBtn.clicked.connect(self.apply_sift)
    
    def setup_subscriptions(self):
        # Subscribe to processing results
        pub.subscribe(self.on_corner_detection_complete, Topics.CORNER_DETECTION_COMPLETE)
        pub.subscribe(self.on_matching_complete, Topics.MATCHING_COMPLETE)
        pub.subscribe(self.on_sift_complete, Topics.SIFT_COMPLETE)
    
    # Corner Detection methods
    def load_corner_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            try:
                # Read and store image
                self.corner_image = cv2.imread(file_path)
                # Display image
                self.display_image(self.corner_image, self.ui.CornerImage)
                # Publish event
                # pub.sendMessage(Topics.LOAD_CORNER_IMAGE, image_path=file_path, image=self.corner_image)
                logging.info(f"Loaded corner image: {file_path}")
            except Exception as e:
                logging.error(f"Error loading corner image: {str(e)}")
    
    def apply_corner_detection(self):
        if self.corner_image is not None:
            method = self.ui.CornerCombo.currentText()
            threshold = self.ui.CornerThresholdSpinBox.value()
            pub.sendMessage(
                Topics.APPLY_CORNER_DETECTION,
                image=self.corner_image,
                method=method,
                threshold=threshold
            )
            logging.info(f"Applied corner detection with {method}, threshold={threshold}")
    
    def on_corner_detection_complete(self, result_image, computation_time):
        self.display_image(result_image, self.ui.CornerOutput)
        self.ui.CornerTime.display(computation_time)
    
    # Matching methods
    def load_matching_image1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image 1", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            try:
                self.matching_image1 = cv2.imread(file_path)
                self.display_image(self.matching_image1, self.ui.MatchingImage1)
                # pub.sendMessage(Topics.LOAD_MATCHING_IMAGE1, image_path=file_path, image=self.matching_image1)
                logging.info(f"Loaded matching image 1: {file_path}")
            except Exception as e:
                logging.error(f"Error loading matching image 1: {str(e)}")
    
    def load_matching_image2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image 2", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            try:
                self.matching_image2 = cv2.imread(file_path)
                self.display_image(self.matching_image2, self.ui.MatchingImage2)
                # pub.sendMessage(Topics.LOAD_MATCHING_IMAGE2, image_path=file_path, image=self.matching_image2)
                logging.info(f"Loaded matching image 2: {file_path}")
            except Exception as e:
                logging.error(f"Error loading matching image 2: {str(e)}")
    
    def apply_matching(self):
        if self.matching_image1 is not None and self.matching_image2 is not None:
            method = self.ui.MatchingCombo.currentText()
            pub.sendMessage(
                Topics.APPLY_MATCHING,
                image1=self.matching_image1,
                image2=self.matching_image2,
                method=method
            )
            logging.info(f"Applied matching with {method}")
    
    def on_matching_complete(self, result_image, computation_time):
        self.display_image(result_image, self.ui.MatchingOutPut)
        self.ui.MatchingTime.display(computation_time)
    
    # SIFT methods
    def load_sift_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            try:
                self.sift_image = cv2.imread(file_path)
                self.display_image(self.sift_image, self.ui.SiftImage)
                # pub.sendMessage(Topics.LOAD_SIFT_IMAGE1, image_path=file_path, image=self.sift_image1)
                # logging.info(f"Loaded SIFT image 1: {file_path}")
            except Exception as e:
                logging.error(f"Error loading SIFT image 1: {str(e)}")

    
    def apply_sift(self):
        if self.sift_image is not None :
            pub.sendMessage(Topics.APPLY_SIFT,image=self.sift_image)
            logging.info("Applied SIFT")
    
    def on_sift_complete(self, result_image, computation_time):
        self.display_image(result_image, self.ui.SiftOutput)
        self.ui.SIFTTime.display(computation_time)
    
    # Utility methods
    def display_image(self, image, label):
        if image is None:
            return
        
        # Convert from BGR (OpenCV) to RGB
        if len(image.shape) == 3:  # Color image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:  # Grayscale image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        h, w = rgb_image.shape[:2]
        
        # Get the available size for the label
        label_width = label.width()
        label_height = label.height()
        
        # Calculate the aspect ratio
        aspect_ratio = w / h
        
        # Determine the size for display
        if label_width / label_height > aspect_ratio:
            display_height = label_height
            display_width = int(display_height * aspect_ratio)
        else:
            display_width = label_width
            display_height = int(display_width / aspect_ratio)
        
        # Create a QImage and then a QPixmap
        qimage = QImage(rgb_image.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        
        # Set the scaled pixmap to the label
        label.setPixmap(pixmap.scaled(display_width, display_height, Qt.AspectRatioMode.KeepAspectRatio))

