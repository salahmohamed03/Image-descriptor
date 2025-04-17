from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import numpy as np
from scipy.ndimage import convolve,gaussian_filter
import cv2

class CornerDetection:
    def __init__(self):
        self.setup_subscriptions()

    def setup_subscriptions(self):
        pub.subscribe(self.on_ApplyCornerDetection, Topics.APPLY_CORNER_DETECTION)

    def on_ApplyCornerDetection(self, image, method, threshold):
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.get_event_loop()
        # Run the CPU-intensive task in a separate thread
        loop.run_in_executor(executor, self.apply, image, method, threshold)
    def apply(self, image, method, threshold):
        start = time.time()

        # Step 0: Ensure the image is in RGB format for drawing
        if len(image.shape) == 2:  # If grayscale, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        result_image = image.copy()  # Keep the original RGB image for drawing

        # Step 1: Call the appropriate method based on the input
        if method.lower() == "harris":
            result_image = self._harris_corner_detection(result_image, threshold)
        else:  
            result_image = self._lambda_corner_detection(result_image, threshold)

        computation_time = time.time() - start
        pub.sendMessage(
            Topics.CORNER_DETECTION_COMPLETE,
            result_image=result_image,
            computation_time=computation_time
        )

    def _harris_corner_detection(self, image, threshold):
        # Convert to grayscale for gradient computations
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Step 1: Compute gradients on the grayscale image
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Apply a light Gaussian blur to reduce noise
        gray_image = gaussian_filter(gray_image.astype(float), sigma=0.5)

        # Compute gradients
        Ix = convolve(gray_image, sobel_x)
        Iy = convolve(gray_image, sobel_y)

        # Compute auto-correlation components
        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        Ixy = Ix * Iy

        # Step 2: Apply Gaussian smoothing to the components
        sigma = 1.5  # Lower sigma for less smoothing, capturing more corners
        Sx2 = gaussian_filter(Ix2, sigma=sigma)
        Sy2 = gaussian_filter(Iy2, sigma=sigma)
        Sxy = gaussian_filter(Ixy, sigma=sigma)

        # Step 3: Compute corner response R
        k = 0.04  # Lower k to make the detector less selective
        det_M = Sx2 * Sy2 - Sxy ** 2
        trace_M = Sx2 + Sy2
        R = det_M - k * (trace_M ** 2)

        # Normalize R for consistent thresholding
        R = (R - R.min()) / (R.max() - R.min() + 1e-6)

        # Step 4: Thresholding and non-maximum suppression
        window_size = 5  # Smaller window for less aggressive suppression
        threshold = threshold if threshold else 0.05  # Lower threshold to capture more corners
        corners = []

        # Find local maxima with a less restrictive approach
        for y in range(window_size//2, R.shape[0] - window_size//2):
            for x in range(window_size//2, R.shape[1] - window_size//2):
                local_window = R[y - window_size//2:y + window_size//2 + 1,
                                 x - window_size//2:x + window_size//2 + 1]
                if R[y, x] == np.max(local_window) and R[y, x] > threshold:
                    corners.append((x, y, R[y, x]))

        # Step 5: Sort corners by response and keep more corners
        corners = sorted(corners, key=lambda x: x[2], reverse=True)
        max_corners = 50  # Increase to allow more corners
        corners = corners[:max_corners]

        # Step 6: Draw the corners on the original RGB image
        for x, y, _ in corners:
            cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)

        return image

    def _lambda_corner_detection(self, image, threshold):
        return image
 
