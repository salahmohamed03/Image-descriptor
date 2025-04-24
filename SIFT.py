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
        self.sift = SIFT()  # Using the provided SIFT implementation
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

        # Use the provided SIFT implementation
        self.sift.build_scale_space(gray_image)
        extrema = self.sift.detect_extrema()
        keypoints = self.sift.localize_keypoints(extrema)
        oriented_keypoints = self.sift.assign_orientations(keypoints, gray_image)
        filtered_keypoints, descriptors = self.sift.compute_descriptors(oriented_keypoints, gray_image)
        
        # Convert keypoints to our expected format
        keypoint_dicts = []
        for kp in filtered_keypoints:
            keypoint_dicts.append({
                'x': kp.pt[0],
                'y': kp.pt[1],
                'octave': 0,  # Not directly available in the new implementation
                'scale': kp.size / 2,
                'value': 0  # Not directly available
            })
        
        # Convert orientations to our expected format
        dominant_orientations = []
        for kp in oriented_keypoints:
            dominant_orientations.append({
                'x': kp.pt[0],
                'y': kp.pt[1],
                'octave': 0,  # Not directly available
                'scale': kp.size / 2,
                'orientation': kp.angle
            })
        
        self.draw_keypoints(image, keypoint_dicts, dominant_orientations)
        print("Keypoints Drawn")

        return {
            "image": image,
            "scale_space": self.sift.scale_space, 
            "keypoints": keypoint_dicts,
            "dominant_orientations": dominant_orientations,
            "descriptors": descriptors
        }
        
    def draw_keypoints(self, image, keypoints, orientations):
        for kp in keypoints:
            x, y = int(round(kp['x'])), int(round(kp['y']))
            
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (x, y), 2, (255, 255, 255), -1)  

class SIFT:
    def __init__(self, sigma=1.6, s=3, num_octaves=4):
        self.sigma = sigma
        self.s = s 
        self.k = 2 ** (1.0 / s) 
        self.num_levels = s + 3 
        self.num_octaves = num_octaves
        self.scale_space = None 

    def build_scale_space(self, image):
        image = image.astype(np.float32)
        self.scale_space = []

        for i in range(self.num_octaves):
            
            if i == 0:
                base_image = image

            else:
                base_image = self.scale_space[i-1][self.s][::2, ::2] 

            octave = []

            for level in range(self.num_levels):

                sigma_of_this_level = self.sigma * (self.k ** level)

                gaussian_kernel_size = int(np.ceil(sigma_of_this_level * 3) * 2 + 1)

                if gaussian_kernel_size % 2 == 0:
                    gaussian_kernel_size += 1

                blurred_image = cv2.GaussianBlur(base_image, (gaussian_kernel_size, gaussian_kernel_size), sigmaX = sigma_of_this_level, sigmaY = sigma_of_this_level)

                octave.append(blurred_image) 

            self.scale_space.append(octave)


    def detect_extrema(self):
        if self.scale_space is None:
            raise ValueError("Scale space has not been constructed. Call build_scale_space first.")
        
        DoG = []

        for octave_index in range(self.num_octaves):

            octave_DoG = []
            
            for level in range(self.num_levels - 1):
                dog = self.scale_space[octave_index][level + 1] - self.scale_space[octave_index][level]
                octave_DoG.append(dog)

            DoG.append(octave_DoG)

        extrema = []

        for octave_index in range(self.num_octaves):
            for level in range(1, self.num_levels - 2):

                dog_prev = DoG[octave_index][level - 1]
                dog_curr = DoG[octave_index][level]
                dog_next = DoG[octave_index][level + 1]

                height, width = dog_curr.shape

                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        val = dog_curr[y, x]
                        neighbors = [
                            dog_curr[y-1, x-1], dog_curr[y-1, x], dog_curr[y-1, x+1],
                            dog_curr[y, x-1],                     dog_curr[y, x+1],
                            dog_curr[y+1, x-1], dog_curr[y+1, x], dog_curr[y+1, x+1],
                            dog_prev[y-1, x-1], dog_prev[y-1, x], dog_prev[y-1, x+1],
                            dog_prev[y, x-1],   dog_prev[y, x],   dog_prev[y, x+1],
                            dog_prev[y+1, x-1], dog_prev[y+1, x], dog_prev[y+1, x+1],
                            dog_next[y-1, x-1], dog_next[y-1, x], dog_next[y-1, x+1],
                            dog_next[y, x-1],   dog_next[y, x],   dog_next[y, x+1],
                            dog_next[y+1, x-1], dog_next[y+1, x], dog_next[y+1, x+1]
                        ]
                        if val > max(neighbors) or val < min(neighbors):
                            extrema.append((octave_index, level, x, y))
                            
        return extrema


    def localize_keypoints(self, extrema, contrast_threshold=0.03, edge_threshold=10):
        
        keypoints = []
        DoG_pyramid = []
        for o in range(self.num_octaves):
            octave_DoG = []
            for m in range(self.num_levels - 1):
                dog = self.scale_space[o][m + 1] - self.scale_space[o][m]
                octave_DoG.append(dog)
            DoG_pyramid.append(octave_DoG)

        for (octave, level, x, y) in extrema:

            dog_prev = DoG_pyramid[octave][level - 1]
            dog_curr = DoG_pyramid[octave][level]
            dog_next = DoG_pyramid[octave][level + 1]

            if x < 1 or y < 1 or x >= dog_curr.shape[1] - 1 or y >= dog_curr.shape[0] - 1:
                continue

            Dx = (dog_curr[y, x+1] - dog_curr[y, x-1]) / 2.0
            Dy = (dog_curr[y+1, x] - dog_curr[y-1, x]) / 2.0
            Ds = (dog_next[y, x] - dog_prev[y, x]) / 2.0
            gradient = np.array([Dx, Dy, Ds])

            Dxx = dog_curr[y, x+1] - 2 * dog_curr[y, x] + dog_curr[y, x-1]
            Dyy = dog_curr[y+1, x] - 2 * dog_curr[y, x] + dog_curr[y-1, x]
            Dss = dog_next[y, x] - 2 * dog_curr[y, x] + dog_prev[y, x]

            Dxy = (dog_curr[y+1, x+1] - dog_curr[y+1, x-1] - dog_curr[y-1, x+1] + dog_curr[y-1, x-1]) / 4.0
            Dxs = (dog_next[y, x+1] - dog_next[y, x-1] - dog_prev[y, x+1] + dog_prev[y, x-1]) / 4.0
            Dys = (dog_next[y+1, x] - dog_next[y-1, x] - dog_prev[y+1, x] + dog_prev[y-1, x]) / 4.0
            hessian = np.array([
                [Dxx, Dxy, Dxs],
                [Dxy, Dyy, Dys],
                [Dxs, Dys, Dss]
            ])

            try:
                offset = -np.linalg.inv(hessian).dot(gradient)
            except np.linalg.LinAlgError:
                continue 

            if np.any(np.abs(offset) > 0.5):
                continue

            x_refined = x + offset[0]
            y_refined = y + offset[1]
            m_refined = level + offset[2]

            D_refined = dog_curr[y, x] + 0.5 * gradient.dot(offset)
            if abs(D_refined) < contrast_threshold:
                continue # Low contrast, discard

            H_2d = np.array([[Dxx, Dxy], [Dxy, Dyy]])
            trace = Dxx + Dyy
            det = Dxx * Dyy - Dxy ** 2
            if det <= 0 or trace ** 2 / det >= (edge_threshold + 1) ** 2 / edge_threshold:
                continue # Edge-like, discard

            # Convert to original image coordinates and sigma
            scale_factor = 2 ** octave
            x_final = x_refined * scale_factor
            y_final = y_refined * scale_factor
            sigma_final = self.sigma * (self.k ** m_refined) * scale_factor

            kp = cv2.KeyPoint(x_final, y_final, sigma_final * 2)
            keypoints.append(kp)
        return keypoints


    def assign_orientations(self, keypoints, image):

        oriented_keypoints = []

        for key_point in keypoints:

            x, y = int(key_point.pt[0]), int(key_point.pt[1])

            sigma = key_point.size / 2

            radius = int(np.ceil(1.5 * sigma))

            if (x - radius < 0 or x + radius >= image.shape[1] or 
                y - radius < 0 or y + radius >= image.shape[0]):
                continue

            patch = image[y-radius : y+radius+1, x-radius : x+radius+1]

            dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)

            magnitude = np.sqrt(dx**2 + dy**2)

            direction = np.arctan2(dy, dx) * 180 / np.pi

            direction = (direction + 360) % 360

            y_coords, x_coords = np.indices(patch.shape)

            center = radius

            gaussian = np.exp(-((x_coords - center)**2 + (y_coords - center)**2) / (2 * sigma**2))

            weights = magnitude * gaussian

            hist = np.zeros(36) 
            
            for i in range(patch.shape[0]):
                for j in range(patch.shape[1]):
                    bin_idx = int(direction[i, j] // 10)
                    hist[bin_idx] += weights[i, j]

            hist_smoothed = np.convolve(hist, [1, 1, 1], mode='same') / 3

            max_idx = np.argmax(hist_smoothed)

            if hist_smoothed[max_idx] == 0:
                continue
            
            angle = float((max_idx * 10 + 5) % 360)
            new_kp = cv2.KeyPoint(float(x), float(y), key_point.size, angle)
            oriented_keypoints.append(new_kp)

        return oriented_keypoints


    def compute_descriptors(self, keypoints, image):
        filtered_keypoints = []
        descriptors = []
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            sigma = kp.size / 2
            orientation = kp.angle
            radius = int(np.ceil(3 * sigma))
            if (x - radius < 0 or x + radius >= image.shape[1] or 
                y - radius < 0 or y + radius >= image.shape[0]):
                continue

            patch = image[y-radius:y+radius+1, x-radius:x+radius+1]
            dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx) * 180 / np.pi
            direction = (direction + 360) % 360

            y_coords, x_coords = np.indices(patch.shape)
            center = radius
            gaussian = np.exp(-((x_coords - center)**2 + (y_coords - center)**2) / (2 * (1.5 * sigma)**2))
            weights = magnitude * gaussian

            direction = (direction - orientation + 360) % 360
            patch_size = patch.shape[0]
            subregion_size = patch_size // 4
            descriptor = []

            for i in range(4):
                for j in range(4):
                    y_start = i * subregion_size
                    y_end = (i + 1) * subregion_size
                    x_start = j * subregion_size
                    x_end = (j + 1) * subregion_size
                    sub_weights = weights[y_start:y_end, x_start:x_end]
                    sub_directions = direction[y_start:y_end, x_start:x_end]
                    hist = np.zeros(8)
                    for sy in range(sub_weights.shape[0]):
                        for sx in range(sub_weights.shape[1]):
                            bin_idx = int(sub_directions[sy, sx] // 45)
                            hist[bin_idx] += sub_weights[sy, sx]
                    descriptor.extend(hist)

            descriptor = np.array(descriptor)
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
                descriptor = np.clip(descriptor, 0, 0.2)
                norm = np.linalg.norm(descriptor)
                if norm > 0:
                    descriptor = descriptor / norm
                    filtered_keypoints.append(kp)
                    descriptors.append(descriptor)

        return filtered_keypoints, np.array(descriptors)