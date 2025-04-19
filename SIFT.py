from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import time
import cv2
import numpy as np

class SIFTProcessor:
    def __init__(self):
        self.NUM_OCTAVES = 4  # Reduced for faster computation
        self.NUM_SCALES = 3
        self.SIGMA = 1.6
        self.CONTRAST_THRESHOLD = 0.03
        self.EDGE_THRESHOLD = 10.0
        self.NUM_BINS = 36
        self.DESCRIPTOR_BINS = 8
        self.DESCRIPTOR_REGIONS = 4
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
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray_image = image.copy().astype(np.float32)
            image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # Build scale-space pyramid and detect keypoints
        scale_space, sigmas = self.build_scale_space(gray_image)
        keypoints = self.detect_keypoints(scale_space, sigmas)
        print(f"Keypoint Detection Done - {len(keypoints)} keypoints")

        # Convert keypoints to dictionary format for compatibility
        keypoint_dicts = [{'x': kp[0], 'y': kp[1], 'octave': kp[2], 
                          'scale': kp[3], 'value': kp[4]} for kp in keypoints]

        # Assign orientations
        orientations = self.assign_orientations(gray_image, keypoints, sigmas)
        print(f"Orientation Assignment Done - {len(orientations)} orientations")

        # Generate descriptors
        descriptors = self.generate_descriptors(gray_image, keypoints, orientations, sigmas)
        print(f"Descriptor Generation Done - {len(descriptors)} descriptors")

        # Draw keypoints on the image
        self.draw_keypoints(image, keypoint_dicts, orientations)

        return {
            "image": image,
            "scale_space": scale_space,
            "keypoints": keypoint_dicts,
            "DOG_pyramids": [],  # Not used in this implementation but included for compatibility
            "dominant_orientations": orientations,
            "descriptors": descriptors
        }

    def build_scale_space(self, image):
        # Double the image size to capture more details in the first octave
        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_LINEAR)

        scale_space = []
        sigmas = []
        k = 2 ** (1.0 / self.NUM_SCALES)

        current_image = image
        for octave in range(self.NUM_OCTAVES):
            octave_images = []
            octave_sigmas = []
            sigma = self.SIGMA

            # Apply initial blur if first octave
            if octave == 0:
                current_image = cv2.GaussianBlur(current_image, (0, 0), sigmaX=sigma)
            
            octave_images.append(current_image)
            octave_sigmas.append(sigma)

            # Generate blurred images for each scale
            for scale in range(1, self.NUM_SCALES + 3):  # Extra scales for DoG
                sigma_diff = np.sqrt((k ** scale * self.SIGMA) ** 2 - (k ** (scale - 1) * self.SIGMA) ** 2)
                current_image = cv2.GaussianBlur(current_image, (0, 0), sigmaX=sigma_diff)
                octave_images.append(current_image)
                octave_sigmas.append(k ** scale * self.SIGMA)

            scale_space.append(octave_images)
            sigmas.append(octave_sigmas)

            # Downsample for the next octave
            current_image = octave_images[self.NUM_SCALES]  # Use the image at the last scale
            current_image = cv2.resize(current_image, (current_image.shape[1] // 2, current_image.shape[0] // 2), 
                                      interpolation=cv2.INTER_NEAREST)

        return scale_space, sigmas

    def detect_keypoints(self, scale_space, sigmas):
        keypoints = []

        for octave in range(self.NUM_OCTAVES):
            dog_images = []
            # Compute Difference of Gaussians
            for scale in range(self.NUM_SCALES + 2):
                dog = scale_space[octave][scale + 1] - scale_space[octave][scale]
                dog_images.append(dog)

            height, width = dog_images[0].shape
            for scale in range(1, self.NUM_SCALES + 1):
                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        val = dog_images[scale][y, x]
                        # Check if it's an extremum
                        is_extremum = True
                        is_max = val > 0
                        is_min = val < 0

                        # Compare with 26 neighbors (3x3x3 - 1)
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                for ds in [-1, 0, 1]:
                                    if dy == 0 and dx == 0 and ds == 0:
                                        continue
                                    neighbor = dog_images[scale + ds][y + dy, x + dx]
                                    if is_max and val <= neighbor:
                                        is_extremum = False
                                        break
                                    if is_min and val >= neighbor:
                                        is_extremum = False
                                        break
                            if not is_extremum:
                                break
                        if not is_extremum:
                            continue

                        # Contrast check
                        if abs(val) < self.CONTRAST_THRESHOLD:
                            continue

                        # Edge response elimination using Hessian
                        Dxx = (dog_images[scale][y, x + 1] + dog_images[scale][y, x - 1] - 2 * val)
                        Dyy = (dog_images[scale][y + 1, x] + dog_images[scale][y - 1, x] - 2 * val)
                        Dxy = ((dog_images[scale][y + 1, x + 1] - dog_images[scale][y + 1, x - 1]) -
                               (dog_images[scale][y - 1, x + 1] - dog_images[scale][y - 1, x - 1])) / 4.0
                        trH = Dxx + Dyy
                        detH = Dxx * Dyy - Dxy * Dxy
                        if detH <= 0:
                            continue
                        curvature_ratio = (trH * trH) / detH
                        if curvature_ratio > (self.EDGE_THRESHOLD + 1) ** 2 / self.EDGE_THRESHOLD:
                            continue

                        # Adjust coordinates for octave
                        adjusted_x = x * (2 ** octave)
                        adjusted_y = y * (2 ** octave)
                        keypoints.append((adjusted_x, adjusted_y, octave, scale, val))

        return keypoints

    def assign_orientations(self, image, keypoints, sigmas):
        orientations = []
        height, width = image.shape

        for kp in keypoints:
            x, y, octave, scale, _ = kp
            sigma = sigmas[octave][scale] * (2 ** octave)

            # Define window for orientation computation
            window_radius = int(3 * sigma)
            if window_radius < 1:
                continue

            x_start = max(0, int(x - window_radius))
            x_end = min(width, int(x + window_radius + 1))
            y_start = max(0, int(y - window_radius))
            y_end = min(height, int(y + window_radius + 1))

            if x_end <= x_start or y_end <= y_start:
                continue

            # Compute gradients
            window = image[y_start:y_end, x_start:x_end]
            dx = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(dx**2 + dy**2)
            ori = np.degrees(np.arctan2(dy, dx)) % 360

            # Gaussian weighting
            yy, xx = np.mgrid[0:window.shape[0], 0:window.shape[1]]
            center_x = x - x_start
            center_y = y - y_start
            gaussian = np.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))
            weighted_mag = mag * gaussian

            # Compute orientation histogram
            hist = np.zeros(self.NUM_BINS)
            bin_size = 360.0 / self.NUM_BINS
            for i in range(window.shape[0]):
                for j in range(window.shape[1]):
                    angle = ori[i, j]
                    bin_idx = int(angle / bin_size) % self.NUM_BINS
                    hist[bin_idx] += weighted_mag[i, j]

            # Smooth histogram
            smoothed_hist = np.convolve(hist, [0.25, 0.5, 0.25], mode='same')

            # Find peaks
            max_peak = np.max(smoothed_hist)
            peaks = np.where(smoothed_hist >= 0.8 * max_peak)[0]
            for peak in peaks:
                left = smoothed_hist[(peak - 1) % self.NUM_BINS]
                center = smoothed_hist[peak]
                right = smoothed_hist[(peak + 1) % self.NUM_BINS]
                if center <= left or center <= right:
                    continue
                offset = 0.5 * (left - right) / (left - 2 * center + right) if (left - 2 * center + right) != 0 else 0
                orientation = (peak + offset) * bin_size
                orientations.append({
                    'x': x,
                    'y': y,
                    'octave': octave,
                    'scale': scale,
                    'orientation': orientation
                })

        return orientations

    def generate_descriptors(self, image, keypoints, orientations, sigmas):
        descriptors = []
        height, width = image.shape

        # Compute gradients for the entire image
        dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(dx**2 + dy**2)
        ori = np.degrees(np.arctan2(dy, dx)) % 360

        # Organize orientations
        orientation_dict = {}
        for orient in orientations:
            x, y = int(orient['x']), int(orient['y'])
            if x not in orientation_dict:
                orientation_dict[x] = {}
            if y not in orientation_dict[x]:
                orientation_dict[x][y] = []
            orientation_dict[x][y].append(orient)

        for kp in keypoints:
            x, y, octave, scale, _ = kp
            sigma = sigmas[octave][scale] * (2 ** octave)

            # Define descriptor window (16x16 at base scale, adjusted for octave)
            window_size = int(16 * (sigma / self.SIGMA))
            if window_size % 2 == 0:
                window_size += 1
            half_window = window_size // 2

            if (x < half_window or x >= width - half_window or
                y < half_window or y >= height - half_window):
                continue

            try:
                kp_orientations = orientation_dict.get(int(x), {}).get(int(y), [])
                if not kp_orientations:
                    continue
            except KeyError:
                continue

            for orient in kp_orientations:
                main_orientation = orient['orientation']
                descriptor = []

                # Extract window
                y_start = int(y - half_window)
                y_end = int(y + half_window + 1)
                x_start = int(x - half_window)
                x_end = int(x + half_window + 1)
                mag_window = mag[y_start:y_end, x_start:x_end]
                ori_window = ori[y_start:y_end, x_start:x_end]

                # Rotate orientations relative to main orientation
                ori_window = (ori_window - main_orientation) % 360

                # Divide into 4x4 sub-regions
                subregion_size = window_size // self.DESCRIPTOR_REGIONS
                for i in range(self.DESCRIPTOR_REGIONS):
                    for j in range(self.DESCRIPTOR_REGIONS):
                        y_sub_start = i * subregion_size
                        y_sub_end = (i + 1) * subregion_size
                        x_sub_start = j * subregion_size
                        x_sub_end = (j + 1) * subregion_size

                        sub_mag = mag_window[y_sub_start:y_sub_end, x_sub_start:x_sub_end]
                        sub_ori = ori_window[y_sub_start:y_sub_end, x_sub_start:x_sub_end]

                        # Gaussian weighting
                        yy, xx = np.mgrid[0:subregion_size, 0:subregion_size]
                        center = subregion_size / 2
                        gaussian = np.exp(-((xx - center)**2 + (yy - center)**2) / (2 * (sigma)**2))
                        weighted_mag = sub_mag * gaussian

                        # 8-bin histogram
                        hist = np.zeros(self.DESCRIPTOR_BINS)
                        bin_size = 360.0 / self.DESCRIPTOR_BINS
                        for m in range(subregion_size):
                            for n in range(subregion_size):
                                angle = sub_ori[m, n]
                                bin_idx = int(angle / bin_size) % self.DESCRIPTOR_BINS
                                hist[bin_idx] += weighted_mag[m, n]

                        descriptor.extend(hist)

                # Normalize descriptor
                descriptor = np.array(descriptor)
                norm = np.linalg.norm(descriptor)
                if norm > 0:
                    descriptor = descriptor / norm
                    descriptor = np.clip(descriptor, 0, 0.2)
                    norm = np.linalg.norm(descriptor)
                    if norm > 0:
                        descriptor = descriptor / norm

                descriptors.append(descriptor)

        return np.array(descriptors)

    def draw_keypoints(self, image, keypoints, orientations):
        for kp in keypoints:
            x, y = int(kp['x']), int(kp['y'])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (x, y), 2, (255, 255, 255), -1)