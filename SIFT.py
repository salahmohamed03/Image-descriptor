from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import cv2
import numpy as np
from numba import njit
from collections import namedtuple

Keypoint = namedtuple('Keypoint', ['x', 'y', 'octave', 'scale', 'value'])

class SIFTProcessor:
    def __init__(self):
        self.NUM_OCTAVES = 8
        self.NUM_SCALES = 10
        self.INITIAL_SIGMA = 1.6
        self.NUM_BINS = 8
        self.NUM_REGIONS = 4
        self.REGION_SIZE = 4
        self.WINDOW_SIZE = self.NUM_REGIONS * self.REGION_SIZE
        self.SIGMA_MULTIPLIER = 1.5
        self.PEAK_RATIO = 0.8
        self.CONTRAST_THRESHOLD = 0.04
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_sift, Topics.APPLY_SIFT)
    
    def on_apply_sift(self, image):
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, self.apply, image)
    
    def apply(self, image):
        print("SIFT STARTED")
        start = time.time()
       
        results = self.calculate_sift(image)
        result_image = results['image']
        
        computation_time = time.time() - start
        
        pub.sendMessage(
            Topics.SIFT_COMPLETE,
            result_image=result_image,
            computation_time=computation_time
        )

    def calculate_sift(self, image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray_image = image.copy().astype(np.float32)
            image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        edgy_image = self.detectEdges(gray_image)
        gray_image = cv2.normalize(edgy_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        scale_space = self.scale_space_construction(gray_image)
        print(f"Scale Space Construction Done - {len(scale_space)} octaves")

        keypoints, DOG_pyramids = self.scale_space_extrema_detection(scale_space)
        print(f"Keypoint Detection Done - {len(keypoints)} keypoints")

        keypoint_dicts = [{'x': kp.x, 'y': kp.y, 'octave': kp.octave, 
                          'scale': kp.scale, 'value': kp.value} for kp in keypoints]
        
        dominant_orientations = self.orientation_assignment(gray_image, keypoint_dicts)
        print(f"Orientation Assignment Done - {len(dominant_orientations)} orientations")

        descriptors = self.build_descriptor(gray_image, keypoint_dicts, dominant_orientations)
        print(f"Descriptor Building Done - {len(descriptors)} descriptors")

        self.draw_keypoints(image, keypoint_dicts, dominant_orientations)
        print("Keypoints Drawn")

        return {
            "image": image,
            "scale_space": scale_space,
            "keypoints": keypoint_dicts,
            "DOG_pyramids": DOG_pyramids,
            "dominant_orientations": dominant_orientations,
            "descriptors": descriptors
        }
        
    def draw_keypoints(self, image, keypoints, orientations):
        for kp in keypoints:
            x, y = int(round(kp['x'])), int(round(kp['y']))
            
            # Draw a small white highlight circle (radius=3, thickness=-1 for filled)
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (x, y), 1, (255, 255, 255), -1)  # White filled circle
                
    def scale_space_construction(self, image):
        scale_space = []
        copy_image = image.copy()
        k = np.sqrt(2)

        for octave in range(self.NUM_OCTAVES):
           octaves = []
           sigma= self.INITIAL_SIGMA
           for scale in range(self.NUM_SCALES):
                current_sigma = sigma * (k ** scale)
                blurred_image = cv2.GaussianBlur(copy_image, (0, 0), sigmaX=current_sigma)
                octaves.append(blurred_image)
           scale_space.append(octaves)
           # Downgrade the image 
           image = cv2.resize(image, (int(
                image.shape[1] / 2), int(image.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
    
        return scale_space
    

    def scale_space_extrema_detection(self, scale_space):
        keypoints = []
        DOG_pyramids = []
        
        # Build DOG Pyramid
        for octave in range(len(scale_space)):
            DOG_octave = []
            for scale in range(len(scale_space[octave]) - 1):
                DOG = scale_space[octave][scale + 1] - scale_space[octave][scale]
                DOG_octave.append(DOG)
            DOG_pyramids.append(DOG_octave)
        
        # Find Local Extrema with contrast threshold
        for octave, DOG_octave in enumerate(DOG_pyramids):
            for scale in range(1, len(DOG_octave) - 1):
                current_DOG = DOG_octave[scale]
                above_DOG = DOG_octave[scale + 1]
                below_DOG = DOG_octave[scale - 1]
                
                octave_keypoints = self._find_extrema_in_octave(
                    current_DOG, above_DOG, below_DOG, octave, scale
                )
                keypoints.extend(octave_keypoints)
        
        return keypoints, DOG_pyramids
    
    @staticmethod
    @njit
    def _find_extrema_in_octave(current_DOG, above_DOG, below_DOG, octave, scale):
        keypoints = []
        height, width = current_DOG.shape
        # Lower threshold to detect more keypoints
        threshold = 0.05  # Reduced from 0.032 (0.8*0.04)
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                current_val = current_DOG[i, j]
                
                # First check if the value is significant enough
                if abs(current_val) < threshold:
                    continue
                    
                is_max = True
                is_min = True
                
                # Check current scale neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        
                        neighbor = current_DOG[i + di, j + dj]
                        if current_val <= neighbor:
                            is_max = False
                        if current_val >= neighbor:
                            is_min = False
                        if not is_max and not is_min:
                            break
                    if not is_max and not is_min:
                        break
                
                # Only check other scales if we might have an extremum
                if is_max or is_min:
                    # Check scale above
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            neighbor = above_DOG[i + di, j + dj]
                            if current_val <= neighbor:
                                is_max = False
                            if current_val >= neighbor:
                                is_min = False
                            if not is_max and not is_min:
                                break
                        if not is_max and not is_min:
                            break
                
                if is_max or is_min:
                    # Check scale below
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            neighbor = below_DOG[i + di, j + dj]
                            if current_val <= neighbor:
                                is_max = False
                            if current_val >= neighbor:
                                is_min = False
                            if not is_max and not is_min:
                                break
                        if not is_max and not is_min:
                            break
                
                if is_max or is_min:
                    keypoints.append(Keypoint(
                        x=j * (2 ** octave),
                        y=i * (2 ** octave),
                        octave=octave,
                        scale=scale,
                        value=current_val
                    ))
        
        return keypoints

    def orientation_assignment(self, image, keypoints):
        orientations = []
        height, width = image.shape
        
        for kp in keypoints:
            x, y = kp['x'], kp['y']
            scale = kp['scale']
            sigma = self.SIGMA_MULTIPLIER * scale
            
            window_size = int(6 * sigma)
            window_size = min(window_size, min(height, width) // 2)
            
            y_start = max(0, int(y - window_size))
            y_end = min(height, int(y + window_size + 1))
            x_start = max(0, int(x - window_size))
            x_end = min(width, int(x + window_size + 1))
            
            if y_end <= y_start or x_end <= x_start or (y_end - y_start) < 3 or (x_end - x_start) < 3:
                continue
                
            window = image[y_start:y_end, x_start:x_end]
            dy = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)
            dx = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
            
            magnitudes = np.sqrt(dx**2 + dy**2)
            orientations_deg = np.degrees(np.arctan2(dy, dx)) % 360
            
            center_y, center_x = window.shape[0] // 2, window.shape[1] // 2
            y_indices, x_indices = np.ogrid[:window.shape[0], :window.shape[1]]
            dist_sq = ((y_indices - center_y)**2 + (x_indices - center_x)**2) / (2 * sigma**2)
            gaussian_weights = np.exp(-dist_sq)
            
            weighted_mags = magnitudes * gaussian_weights
            
            bin_indices = (orientations_deg * self.NUM_BINS / 360).astype(int)
            histogram = np.bincount(bin_indices.ravel(), weights=weighted_mags.ravel(), minlength=self.NUM_BINS)
            
            histogram = np.roll(histogram, 1) + histogram + np.roll(histogram, -1)
            
            max_peak = np.max(histogram)
            peak_indices = np.where(histogram >= max_peak * self.PEAK_RATIO)[0]
            
            for peak_idx in peak_indices:
                orientation = 360.0 * peak_idx / self.NUM_BINS
                orientations.append({
                    'x': x,
                    'y': y,
                    'octave': kp['octave'],
                    'scale': scale,
                    'orientation': orientation
                })
                
        return orientations

    def build_descriptor(self, image, keypoints, dominant_orientations):
        descriptors = []
        height, width = image.shape
        
        for kp in dominant_orientations:
            x, y = kp['x'], kp['y']
            orientation = kp['orientation']
            
            if (x < self.WINDOW_SIZE/2 or x > width - self.WINDOW_SIZE/2 or 
                y < self.WINDOW_SIZE/2 or y > height - self.WINDOW_SIZE/2):
                continue
            
            y_start = int(y - self.WINDOW_SIZE/2)
            y_end = int(y + self.WINDOW_SIZE/2)
            x_start = int(x - self.WINDOW_SIZE/2)
            x_end = int(x + self.WINDOW_SIZE/2)
            
            window = image[y_start:y_end, x_start:x_end]
            dy = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)
            dx = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
            
            magnitudes = np.sqrt(dx**2 + dy**2)
            orientations = (np.degrees(np.arctan2(dy, dx)) - orientation) % 360
            
            descriptor = np.zeros((self.NUM_REGIONS * self.NUM_REGIONS * self.NUM_BINS))
            
            for i in range(self.NUM_REGIONS):
                for j in range(self.NUM_REGIONS):
                    region_y = i * self.REGION_SIZE
                    region_x = j * self.REGION_SIZE
                    region_mags = magnitudes[region_y:region_y+self.REGION_SIZE, region_x:region_x+self.REGION_SIZE]
                    region_oris = orientations[region_y:region_y+self.REGION_SIZE, region_x:region_x+self.REGION_SIZE]
                    
                    bin_indices = (region_oris * self.NUM_BINS / 360).astype(int)
                    hist = np.bincount(bin_indices.ravel(), weights=region_mags.ravel(), minlength=self.NUM_BINS)
                    
                    descriptor[(i * self.NUM_REGIONS + j) * self.NUM_BINS:
                              (i * self.NUM_REGIONS + j + 1) * self.NUM_BINS] = hist
            
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
            
            descriptor[descriptor > 0.2] = 0.2
            
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
            
            descriptors.append({
                'x': x,
                'y': y,
                'descriptor': descriptor,
                'orientation': orientation,
                'scale': kp['scale'],
                'octave': kp['octave']
            })
            
        return descriptors
    

    def getDescriptor(self, image):
        result = self.calculate_sift(image)
        return (result['Keypoint'],  result['descriptors'])
    

    def detectEdges(self, gray):
        rows, cols = gray.shape

        ft_components = np.fft.fft2(gray)
        ft_components = np.fft.fftshift(ft_components)

        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel X
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Sobel Y

        Kx_padded = np.zeros_like(gray)
        Ky_padded = np.zeros_like(gray)

        kh, kw = Kx.shape
        Kx_padded[:kh, :kw] = Kx
        Ky_padded[:kh, :kw] = Ky

        Kx_fft = np.fft.fft2(Kx_padded, s=gray.shape)
        Ky_fft = np.fft.fft2(Ky_padded, s=gray.shape)

        Kx_fft = np.fft.fftshift(Kx_fft)
        Ky_fft = np.fft.fftshift(Ky_fft)

        Gx_fft = ft_components * Kx_fft
        Gy_fft = ft_components * Ky_fft

        x_edge_image = np.fft.ifft2(Gx_fft).real
        y_edge_image = np.fft.ifft2(Gy_fft).real
        filtered_image = np.sqrt(x_edge_image**2 + y_edge_image**2)

        x_edge_image = cv2.normalize(np.abs(x_edge_image), None, 0, 255, cv2.NORM_MINMAX)
        y_edge_image = cv2.normalize(np.abs(y_edge_image), None, 0, 255, cv2.NORM_MINMAX)
        filtered_image = cv2.normalize(np.sqrt(x_edge_image**2 + y_edge_image**2), None, 0, 255, cv2.NORM_MINMAX)

        return filtered_image