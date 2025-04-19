from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import logging
import time
import cv2
import numpy as np

# Import your custom SIFT processor
from SIFT import SIFTProcessor

class MatchingProcessor:
    def __init__(self):
        self.setup_subscriptions()
        self.sift_processor = SIFTProcessor()   
        logging.info("Matching processor initialized")
    
    def setup_subscriptions(self):
        pub.subscribe(self.on_apply_matching, Topics.APPLY_MATCHING)
    
    def on_apply_matching(self, image1, image2, method):
        executor = concurrent.futures.ThreadPoolExecutor()
        loop = asyncio.get_event_loop()
        # Run the CPU-intensive task in a separate thread
        loop.run_in_executor(executor, self.apply, image1, image2, method)
    
    def apply(self, image1, image2, method):
        print(f"MATCHING STARTED using {method}")
        start = time.time()

        # processing both images with our sift
        results1 = self.sift_processor.calculate_sift(image1)
        results2 = self.sift_processor.calculate_sift(image2)
        
        keypoints1 = results1['keypoints']
        descriptors1 = [kp['descriptor'] for kp in results1['descriptors']] # extracting the descriptor part from each dictionary
        
        keypoints2 = results2['keypoints']
        descriptors2 = [kp['descriptor'] for kp in results2['descriptors']]
        
        if not descriptors1 or not descriptors2:
            print("No descriptors found in one or both images")
            result_image = np.hstack((image1, image2))
            computation_time = time.time() - start
            pub.sendMessage(
                Topics.MATCHING_COMPLETE,
                result_image=result_image,
                computation_time=computation_time
            )
            return
        
        # converting to np arrays for easier manipulation
        descriptors1 = np.array(descriptors1)
        descriptors2 = np.array(descriptors2)
        
        print(f"Found {len(keypoints1)} keypoints in image 1 and {len(keypoints2)} keypoints in image 2")
        
        if method == "SSD":
            matches = self.match_features_ssd(descriptors1, descriptors2)
        elif method == "NCC":
            matches = self.match_features_ncc(descriptors1, descriptors2)
        else:
            # Fallback to SSD
            matches = self.match_features_ssd(descriptors1, descriptors2)
        
        print(f"Found {len(matches)} matches using {method}")
        
        
        matches = sorted(matches, key=lambda x: x[2]) # soring based on the third element which is distance  
        
        num_matches = min(10    , len(matches))
        best_matches = matches[:num_matches]    
        
        result_image = self.draw_matches(image1, image2, keypoints1, keypoints2, best_matches)
        
        computation_time = time.time() - start
        print(f"MATCHING COMPLETED in {computation_time:.2f} seconds")
        
        pub.sendMessage(
            Topics.MATCHING_COMPLETE,
            result_image=result_image,
            computation_time=computation_time
        )
    
    def match_features_ssd(self, descriptors1, descriptors2):
        
        print('we are in SSD')

        matches = []
        
        # we are here matching each descriptor in the first image with all descriptors in the second image
        for i, desc1 in enumerate(descriptors1):
            min_dist = float('inf')
            min_idx = -1
            
            # comparing with every descriptor in the second image
            for j, desc2 in enumerate(descriptors2):
                
                ssd = np.sum((desc1 - desc2) ** 2)
                
                if ssd < min_dist:
                    min_dist = ssd
                    min_idx = j
            
            matches.append((i, min_idx, min_dist))
        
        return matches
    
    def match_features_ncc(self, descriptors1, descriptors2):
         
        matches = []
        
        # For each descriptor in the first image
        for i, desc1 in enumerate(descriptors1):
            max_corr = -float('inf')
            max_idx = -1
            
            desc1_mean = np.mean(desc1)
            
            # comparing with every descriptor in the second image
            for j, desc2 in enumerate(descriptors2):
                
                desc2_mean = np.mean(desc2)
                
                numerator = np.sum((desc1 - desc1_mean) * (desc2 - desc2_mean))
                denominator = np.sqrt(np.sum((desc1 - desc1_mean)**2) * np.sum((desc2 - desc2_mean)**2))
                
                if denominator != 0:
                    ncc = numerator / denominator
                else:
                    ncc = -1  # invalid correlation
                
                if ncc > max_corr: # higher is better (correlation close to 1 = best match)
                    max_corr = ncc
                    max_idx = j
            
            # added negative to NCC so that lower values are better (like SSD)
            matches.append((i, max_idx, -max_corr))
        
        return matches
    
    def draw_matches(self, img1, img2, keypoints1, keypoints2, matches):
  
        # making a new image that has the two images side by side
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]
        
        out_img = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8') # max height, total width
        
        # place the first image to the left
        if len(img1.shape) == 3: # chevkinfg if the image is colored or grayscale
            out_img[:rows1, :cols1, :] = img1
        else:
            out_img[:rows1, :cols1, :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            
        if len(img2.shape) == 3:
            out_img[:rows2, cols1:cols1+cols2, :] = img2
        else:
            out_img[:rows2, cols1:cols1+cols2, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            
        # drawing the matches
        for idx1, idx2, _ in matches:
            
            kp1 = keypoints1[idx1]
            kp2 = keypoints2[idx2]
            
            x1, y1 = int(kp1['x']), int(kp1['y'])
            x2, y2 = int(kp2['x']), int(kp2['y'])

            cv2.circle(out_img, (x1, y1), 4, (0, 255, 0), 1)
            cv2.circle(out_img, (x2 + cols1, y2), 4, (0, 255, 0), 1)

            random_color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.line(out_img, (x1, y1), (x2 + cols1, y2), random_color, thickness= 3)
        
        return out_img