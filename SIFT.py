from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import cv2
import numpy as np
from numba import njit
from collections import namedtuple
from math import exp, sqrt, pi
from numpy.linalg import norm
from sift_processor import _find_extrema_in_octave


Keypoint = namedtuple('Keypoint', ['x', 'y', 'octave', 'scale', 'value'])

class SIFTProcessor:
    def __init__(self):
        self.NUM_OCTAVES = 4
        self.NUM_SCALES = 5  # Need at least 3 scales per octave for DoG
        self.INITIAL_SIGMA = 1.6
        self.NUM_BINS = 36  # For orientation assignment
        self.CONTRAST_THRESHOLD = 0.04
        self.EDGE_RATIO = 10.0
        self.SIGMA_MULTIPLIER = 1.5
        self.PEAK_RATIO = 0.8
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

        # Generate base image by doubling size and blurring
        base_image = cv2.resize(gray_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        base_image = cv2.GaussianBlur(base_image, (0, 0), sigmaX=sqrt(self.INITIAL_SIGMA**2 - 0.5**2))
        
        scale_space = self.scale_space_construction(base_image)
        print(f"Scale Space Construction Done - {len(scale_space)} octaves")

        keypoints, DOG_pyramids = self.scale_space_extrema_detection(scale_space)
        print(f"Keypoint Detection Done - {len(keypoints)} keypoints")

        # Convert keypoints to list of dicts
        keypoint_dicts = [{'x': kp.x, 'y': kp.y, 'octave': kp.octave, 
                         'scale': kp.scale, 'value': kp.value} for kp in keypoints]
        
        dominant_orientations = self.orientation_assignment(scale_space, keypoint_dicts)
        print(f"Orientation Assignment Done - {len(dominant_orientations)} orientations")

        descriptors = self.build_descriptor(scale_space, keypoint_dicts, dominant_orientations)
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
            cv2.circle(image, (x, y), 3, (0, 255, 0), 1)
            for orientation in orientations:
                if orientation['x'] == kp['x'] and orientation['y'] == kp['y']:
                    angle = orientation['orientation']
                    length = 10
                    end_x = int(x + length * np.cos(np.deg2rad(angle)))
                    end_y = int(y + length * np.sin(np.deg2rad(angle)))
                    cv2.line(image, (x, y), (end_x, end_y), (0, 0, 255), 1)
                
    def scale_space_construction(self, image):
        scale_space = []
        k = 2 ** (1.0 / (self.NUM_SCALES - 3))  # Scale multiplicative factor
        
        # Generate Gaussian kernels for each scale
        gaussian_kernels = [0] * self.NUM_SCALES
        gaussian_kernels[0] = self.INITIAL_SIGMA
        for s in range(1, self.NUM_SCALES):
            sigma_prev = (k ** (s - 1)) * self.INITIAL_SIGMA
            sigma_total = k * sigma_prev
            gaussian_kernels[s] = sqrt(sigma_total ** 2 - sigma_prev ** 2)
        
        for octave in range(self.NUM_OCTAVES):
            octave_images = []
            octave_images.append(image)  # First image in octave
            
            # Apply Gaussian blur for each scale
            for s in range(1, self.NUM_SCALES):
                image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernels[s])
                octave_images.append(image)
            
            scale_space.append(octave_images)
            
            # Downsample the image for next octave
            if octave < self.NUM_OCTAVES - 1:
                image = cv2.resize(octave_images[-3], 
                                 (int(image.shape[1]/2), int(image.shape[0]/2)), 
                                 interpolation=cv2.INTER_NEAREST)
        
        return scale_space
        
    def generate_DOG_Pyramid(self, scale_space):
        DOG_pyramids = []
        for octave in scale_space:
            octave_DOG = []
            for s in range(len(octave) - 1):
                DOG = octave[s + 1] - octave[s]
                octave_DOG.append(DOG)
            DOG_pyramids.append(octave_DOG)
        return DOG_pyramids
    
    def scale_space_extrema_detection(self, scale_space):
        keypoints = []
        DOG_pyramids = self.generate_DOG_Pyramid(scale_space)
        
        for octave_idx, dog_octave in enumerate(DOG_pyramids):
            for s in range(1, len(dog_octave) - 1):
                octave_keypoints = _find_extrema_in_octave(
                    dog_octave[s],   # current
                    dog_octave[s+1], # above 
                    dog_octave[s-1], # below
                    octave_idx, s,
                    self.CONTRAST_THRESHOLD
                )
                keypoints.extend(octave_keypoints)
        
        return keypoints, DOG_pyramids

    def orientation_assignment(self, scale_space, keypoints):
        orientations = []
        
        for kp in keypoints:
            octave = kp['octave']
            scale = kp['scale']
            x = int(round(kp['x'] / (2 ** octave)))
            y = int(round(kp['y'] / (2 ** octave)))
            
            gaussian_image = scale_space[octave][scale]
            height, width = gaussian_image.shape
            
            if x < 1 or x >= width - 1 or y < 1 or y >= height - 1:
                continue
                
            # Calculate gradient magnitude and orientation
            dx = gaussian_image[y, x+1] - gaussian_image[y, x-1]
            dy = gaussian_image[y-1, x] - gaussian_image[y+1, x]
            mag = sqrt(dx*dx + dy*dy)
            ori = np.rad2deg(np.arctan2(dy, dx)) % 360
            
            # Create orientation histogram
            hist = np.zeros(self.NUM_BINS)
            radius = int(round(3 * 1.5 * kp['scale']))
            weight_factor = -0.5 / (1.5 * kp['scale']) ** 2
            
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if y + i < 1 or y + i >= height - 1 or x + j < 1 or x + j >= width - 1:
                        continue
                        
                    dx = gaussian_image[y+i, x+j+1] - gaussian_image[y+i, x+j-1]
                    dy = gaussian_image[y+i-1, x+j] - gaussian_image[y+i+1, x+j]
                    mag = sqrt(dx*dx + dy*dy)
                    ori = np.rad2deg(np.arctan2(dy, dx)) % 360
                    weight = exp(weight_factor * (i*i + j*j))
                    
                    bin = int(round(ori * self.NUM_BINS / 360)) % self.NUM_BINS
                    hist[bin] += weight * mag
            
            # Smooth the histogram
            for _ in range(6):
                hist = (np.roll(hist, 1) + hist + np.roll(hist, -1)) / 3
                
            # Find peaks in histogram
            max_mag = np.max(hist)
            for bin in range(self.NUM_BINS):
                if hist[bin] < max_mag * self.PEAK_RATIO:
                    continue
                    
                # Quadratic interpolation for peak position
                left = hist[(bin - 1) % self.NUM_BINS]
                center = hist[bin]
                right = hist[(bin + 1) % self.NUM_BINS]
                peak_offset = 0.5 * (left - right) / (left - 2*center + right)
                ori = (bin + peak_offset) * (360 / self.NUM_BINS)
                
                orientations.append({
                    'x': kp['x'],
                    'y': kp['y'],
                    'octave': kp['octave'],
                    'scale': kp['scale'],
                    'orientation': ori % 360
                })
                
        return orientations

    def build_descriptor(self, scale_space, keypoints, orientations):
        descriptors = []
        
        for kp, ori in zip(keypoints, orientations):
            octave = kp['octave']
            scale = kp['scale']
            x = int(round(kp['x'] / (2 ** octave)))
            y = int(round(kp['y'] / (2 ** octave)))
            angle = -np.deg2rad(ori['orientation'])
            
            gaussian_image = scale_space[octave][scale]
            height, width = gaussian_image.shape
            
            # Calculate gradients
            dx = np.zeros((16, 16))
            dy = np.zeros((16, 16))
            mag = np.zeros((16, 16))
            ori = np.zeros((16, 16))
            
            for i in range(-8, 8):
                for j in range(-8, 8):
                    if y + i < 1 or y + i >= height - 1 or x + j < 1 or x + j >= width - 1:
                        continue
                        
                    dx_val = gaussian_image[y+i, x+j+1] - gaussian_image[y+i, x+j-1]
                    dy_val = gaussian_image[y+i-1, x+j] - gaussian_image[y+i+1, x+j]
                    mag_val = sqrt(dx_val*dx_val + dy_val*dy_val)
                    ori_val = (np.rad2deg(np.arctan2(dy_val, dx_val)) - angle) % 360
                    
                    dx[i+8, j+8] = dx_val
                    dy[i+8, j+8] = dy_val
                    mag[i+8, j+8] = mag_val
                    ori[i+8, j+8] = ori_val
            
            # Create descriptor
            descriptor = np.zeros(128)  # 4x4 regions, 8 bins each
            hist_width = 4  # 16/4
            
            for i in range(4):
                for j in range(4):
                    hist = np.zeros(8)
                    
                    for ii in range(4):
                        for jj in range(4):
                            x_idx = i * 4 + ii
                            y_idx = j * 4 + jj
                            
                            weight = exp(-((x_idx-7.5)**2 + (y_idx-7.5)**2) / (2 * (hist_width * 1.5)**2))
                            bin = int(round(ori[x_idx, y_idx] * 8 / 360)) % 8
                            hist[bin] += weight * mag[x_idx, y_idx]
                    
                    descriptor[(i*4 + j)*8 : (i*4 + j)*8 + 8] = hist
            
            # Normalize descriptor
            descriptor = descriptor / norm(descriptor)
            descriptor = np.clip(descriptor, 0, 0.2)  # Clip large values
            descriptor = descriptor / norm(descriptor)  # Renormalize
            
            descriptors.append(descriptor)
            
        return np.array(descriptors, dtype=np.float32)