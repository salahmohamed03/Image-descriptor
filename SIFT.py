from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import cv2
import numpy as np


class SIFTProcessor:
    def __init__(self):
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_sift, Topics.APPLY_SIFT)
    
    def on_apply_sift(self, image):
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.get_event_loop()
        # Run the CPU-intensive task in a separate thread
        loop.run_in_executor(executor, self.apply, image)
    
    def apply(self, image):
        print("SIFT STARTED")
        start = time.time()
        # write the main code here
        """
            SIFT STEPS
            1. Convert image to grayscale
            2. Apply Gaussian smoothing
            3. Compute gradients in x and y directions
            4. Compute gradient magnitude and orientation
            5. Build Gaussian pyramid
            6. Compute DoG for each octave
            7. Find keypoints and descriptors   

            Or  we can say 
            1- Scale  Space  Construction
            2- Keypoint Detection
            3- Orientation Assignment
            4- Keypoint Descriptor
            5- Reyturn the image with keypoints and descriptors
            """ 

        # Convert to grayscale if the image is not already in grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Space Space Construction
        scale_space = self.scale_space_construction(image)
        print(scale_space.shape)
        
        result_image = None
        ###

        computation_time = time.time() - start
        
        # Send the result back through PubSub
        pub.sendMessage(
            Topics.SIFT_COMPLETE,
            result_image=result_image,
            computation_time=computation_time
        )
            
    def scale_space_construction(self, image):
        
        # First Initalize our used parameters
        NUM_OF_OCTAVES = 4
        NUM_OF_SCALES = 4
        INITIAL_SIGMA = 1.6
        k = 2 ** (1 / NUM_OF_SCALES)
        
        image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        sigma = np.sqrt(INITIAL_SIGMA**2 - 0.5**2)
        image_copy = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)

        SCALE_SPACE = []

        for octave in range(NUM_OF_OCTAVES):
            # Reseting Factor
            octave = []
            current_sigma = INITIAL_SIGMA

            for scale in range(NUM_OF_SCALES):
                # Apply the gaussian filter
                if sigma == 0:
                    blurred_image = image_copy
                else:
                    # Update the sigma for the current scale depending on the previous one
                    sigma = np.sqrt(current_sigma * (k ** scale) - current_sigma*(k * scale -1))
                    blurred_image = cv2.GaussianBlur(image_copy, (0, 0), sigmaX=sigma)
                octave.append(blurred_image)
                # append the scale image to the octave list
            SCALE_SPACE.append(octave)
            # Downgrade the image size to start new octave
            image_copy = image_copy[::2, ::2]
            current_sigma = current_sigma * 2

        return SCALE_SPACE

    def scale_space_keypoint_detection(self, image):
        pass

    def orientation_assignment(self, image):
        pass

    def build_descriptor(self, image):
        pass







    # def fourier_transform(self, image):
    #     print("FOURIER TRANSFORM STARTED")

    #     if image is None or image.size == 0:
    #         print("ERROR: Image is empty or None")
    #         return None

    #     image = image.astype(np.float32)

    #     try:
    #         key = hash(image.tobytes())
    #         if key in Images().cache:
    #             ft_components = Images().cache[key]
    #         else:
    #             ft_components = np.fft.fft2(image)
    #             ft_components = np.fft.fftshift(ft_components) 
    #             Images().cache[key] = ft_components
            
    #         ft_magnitude = np.log(np.abs(ft_components) + 1)  
    #         ft_phase = np.angle(ft_components)

    #         results = {
    #             "ft_magnitude": ft_magnitude,
    #             "ft_phase": ft_phase,
    #             "ft_components": ft_components  
    #         }

    #         print("FOURIER TRANSFORM COMPLETE")
    #         return results
        
    #     except Exception as e:
    #         print(f"ERROR in FFT2: {e}")
    #         return None


    # def detect_edges_sync(self, filter):
    #     images = Images()
    #     image = copy(images.image1.image_data)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    #     rows, cols = gray.shape
    #     print(f"Image shape: {rows}x{cols}")

    #     # Compute Fourier Transform
    #     ft_data = self.fourier_transform(gray)
    #     if ft_data is None:
    #         print("Error: Fourier transform failed.")
    #         return
        
    #     ft_components = ft_data["ft_components"] 

    #     # Choose Edge Detection Filter
    #     if filter == "Sobel":
    #         Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel X
    #         Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Sobel Y
    #     elif filter == "Prewitt":
    #         Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    #         Ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    #     elif filter == "Roberts":
    #         Kx = np.array([[1, 0], [0, -1]])
    #         Ky = np.array([[0, 1], [-1, 0]])
    #     else:
    #         print("Applying Canny filter instead")

    #         # Apply Gaussian Blur to separate horizontal/vertical edges
    #         blurred_x = cv2.GaussianBlur(gray, (1, 5), 0)  # Blur in Y direction
    #         blurred_y = cv2.GaussianBlur(gray, (5, 1), 0)  # Blur in X direction

    #         x_edge_image = cv2.Canny(blurred_x.astype(np.uint8), 50, 150)
    #         y_edge_image = cv2.Canny(blurred_y.astype(np.uint8), 50, 150)
    #         filtered_image = cv2.Canny(gray.astype(np.uint8), 50, 150)

    #         # Display results
    #         results = {
    #             "x_edges": x_edge_image,
    #             "y_edges": y_edge_image,
    #             "filtered_image": filtered_image
    #         }
    #         images.output1 = self.convert_to_displayable(results["x_edges"])
    #         images.output2 = self.convert_to_displayable(results["y_edges"])
    #         images.output3 = self.convert_to_displayable(results["filtered_image"])
    #         pub.sendMessage("update display")
    #         return 

    #     # Zero-Pad Kernels for Fourier Domain Convolution
    #     Kx_padded = np.zeros_like(gray)
    #     Ky_padded = np.zeros_like(gray)

    #     kh, kw = Kx.shape
    #     Kx_padded[:kh, :kw] = Kx
    #     Ky_padded[:kh, :kw] = Ky

    #     # Compute FFT of the Kernels
    #     key = hash(Kx_padded.tobytes())
    #     if key in Images().cache:
    #         Kx_fft = Images().cache[key]
    #     else:
    #         Kx_fft = np.fft.fft2(Kx_padded)
    #         Images().cache[key] = Kx_fft

    #     key = hash(Ky_padded.tobytes())
    #     if key in Images().cache:
    #         Ky_fft = Images().cache[key]
    #     else:
    #         Ky_fft = np.fft.fft2(Ky_padded)
    #         Images().cache[key] = Ky_fft

    #     # Shift for Proper Convolution
    #     Kx_fft = np.fft.fftshift(Kx_fft)
    #     Ky_fft = np.fft.fftshift(Ky_fft)

    #     # Apply Edge Detection in Fourier Domain
    #     Gx_fft = ft_components * Kx_fft
    #     Gy_fft = ft_components * Ky_fft

    #     x_edge_image = np.fft.ifft2(Gx_fft).real
    #     y_edge_image = np.fft.ifft2(Gy_fft).real
    #     filtered_image = np.sqrt(x_edge_image**2 + y_edge_image**2)

    #     # **Normalize Edge Maps to Enhance Visibility**
    #     x_edge_image = cv2.normalize(np.abs(x_edge_image), None, 0, 255, cv2.NORM_MINMAX)
    #     y_edge_image = cv2.normalize(np.abs(y_edge_image), None, 0, 255, cv2.NORM_MINMAX)
    #     filtered_image = cv2.normalize(np.sqrt(x_edge_image**2 + y_edge_image**2), None, 0, 255, cv2.NORM_MINMAX)


    #     print("Filtering complete")

    #     # Store and Display Results
    #     results = {
    #         "x_edges": x_edge_image,
    #         "y_edges": y_edge_image,
    #         "filtered_image": filtered_image
    #     }
    #     images.output1 = self.convert_to_displayable(results["x_edges"])
    #     images.output2 = self.convert_to_displayable(results["y_edges"])
    #     images.output3 = self.convert_to_displayable(results["filtered_image"])

    #     pub.sendMessage("update display")
    #     print("Edge detection complete, results published")
