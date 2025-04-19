from message_types import Topics
from pubsub import pub
import concurrent.futures
import asyncio
import logging
import time
import cv2
import numpy as np

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
        loop.run_in_executor(executor, self.apply, image1, image2, method)
    
    def apply(self, image1, image2, method):
        print(f"MATCHING STARTED using {method}")
        start = time.time()

        results1 = self.sift_processor.calculate_sift(image1)
        results2 = self.sift_processor.calculate_sift(image2)
        print("Got the results")
        
        keypoints1 = results1['keypoints']
        descriptors1 = results1['descriptors']
        print(f"Got {len(keypoints1)} keypoints and {len(descriptors1)} descriptors for image 1")
        
        keypoints2 = results2['keypoints']
        descriptors2 = results2['descriptors']
        print(f"Got {len(keypoints2)} keypoints and {len(descriptors2)} descriptors for image 2")

        if (descriptors1 is None or descriptors2 is None or 
            len(descriptors1) == 0 or len(descriptors2) == 0 or
            len(keypoints1) == 0 or len(keypoints2) == 0):
            print("No valid descriptors or keypoints found in one or both images")
            result_image = np.hstack((image1, image2))
            computation_time = time.time() - start
            pub.sendMessage(
                Topics.MATCHING_COMPLETE,
                result_image=result_image,
                computation_time=computation_time
            )
            return
        
        descriptors1 = np.array(descriptors1)
        descriptors2 = np.array(descriptors2)
        
        print(f"Found {len(keypoints1)} keypoints in image 1 and {len(keypoints2)} keypoints in image 2")
        
        # Sort keypoints by response (value) and limit to top 500 to reduce noise
        keypoints1 = sorted(keypoints1, key=lambda kp: abs(kp['value']), reverse=True)[:500]
        descriptors1 = descriptors1[:500]
        keypoints2 = sorted(keypoints2, key=lambda kp: abs(kp['value']), reverse=True)[:500]
        descriptors2 = descriptors2[:500]
        print(f"Limited to {len(keypoints1)} keypoints in image 1 and {len(keypoints2)} keypoints in image 2")
        
        if method == "SSD":
            matches = self.match_features_ssd(descriptors1, descriptors2, len(keypoints1), len(keypoints2))
        elif method == "NCC":
            matches = self.match_features_ncc(descriptors1, descriptors2, len(keypoints1), len(keypoints2))
        else:
            print(f"Unknown method {method}, falling back to SSD")
            matches = self.match_features_ssd(descriptors1, descriptors2, len(keypoints1), len(keypoints2))
        
        print(f"Found {len(matches)} matches using {method}")
        
        matches = sorted(matches, key=lambda x: x[2])
        
        num_matches = min(10, len(matches))
        best_matches = []
        for match in matches[:num_matches]:
            idx1, idx2, dist = match
            if idx1 < len(keypoints1) and idx2 < len(keypoints2):
                best_matches.append(match)
        
        print(f"Selected {len(best_matches)} valid best matches")
        
        result_image = self.draw_matches(image1, image2, keypoints1, keypoints2, best_matches)
        
        computation_time = time.time() - start
        print(f"MATCHING COMPLETED in {computation_time:.2f} seconds")
        
        pub.sendMessage(
            Topics.MATCHING_COMPLETE,
            result_image=result_image,
            computation_time=computation_time
        )
    
    def match_features_ssd(self, descriptors1, descriptors2, len_keypoints1, len_keypoints2):
        print('Matching with SSD')
        matches = []
        
        for i, desc1 in enumerate(descriptors1):
            if i >= len_keypoints1:
                continue
            best_dist = float('inf')
            second_best_dist = float('inf')
            best_idx = -1
            
            for j, desc2 in enumerate(descriptors2):
                if j >= len_keypoints2:
                    continue
                ssd = np.sum((desc1 - desc2) ** 2)
                if ssd < best_dist:
                    second_best_dist = best_dist
                    best_dist = ssd
                    best_idx = j
                elif ssd < second_best_dist:
                    second_best_dist = ssd
            
            if best_idx != -1 and second_best_dist != 0:
                ratio = best_dist / second_best_dist
                if ratio < 0.8:  # Ratio test
                    matches.append((i, best_idx, best_dist))
        
        print('Completed SSD matching')
        return matches
    
    def match_features_ncc(self, descriptors1, descriptors2, len_keypoints1, len_keypoints2):
        print('Matching with NCC')
        matches = []
        
        for i, desc1 in enumerate(descriptors1):
            if i >= len_keypoints1:
                continue
            best_corr = -float('inf')
            second_best_corr = -float('inf')
            best_idx = -1
            desc1_mean = np.mean(desc1)
            
            for j, desc2 in enumerate(descriptors2):
                if j >= len_keypoints2:
                    continue
                desc2_mean = np.mean(desc2)
                numerator = np.sum((desc1 - desc1_mean) * (desc2 - desc2_mean))
                denominator = np.sqrt(np.sum((desc1 - desc1_mean)**2) * np.sum((desc2 - desc2_mean)**2))
                
                if denominator != 0:
                    ncc = numerator / denominator
                else:
                    ncc = -1
                
                if ncc > best_corr:
                    second_best_corr = best_corr
                    best_corr = ncc
                    best_idx = j
                elif ncc > second_best_corr:
                    second_best_corr = ncc
            
            if best_idx != -1 and second_best_corr != -float('inf'):
                ratio = (1 - best_corr) / (1 - second_best_corr) if (1 - second_best_corr) != 0 else float('inf')
                if ratio < 0.8:  # Ratio test
                    matches.append((i, best_idx, -best_corr))
        
        print('Completed NCC matching')
        return matches
    
    def draw_matches(self, img1, img2, keypoints1, keypoints2, matches):
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]
        
        out_img = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')
        
        if len(img1.shape) == 3:
            out_img[:rows1, :cols1, :] = img1
        else:
            out_img[:rows1, :cols1, :] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            
        if len(img2.shape) == 3:
            out_img[:rows2, cols1:cols1+cols2, :] = img2
        else:
            out_img[:rows2, cols1:cols1+cols2, :] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            
        for idx1, idx2, _ in matches:
            if idx1 >= len(keypoints1) or idx2 >= len(keypoints2):
                print(f"Skipping invalid match: idx1={idx1}, idx2={idx2}")
                continue
            
            kp1 = keypoints1[idx1]
            kp2 = keypoints2[idx2]
            
            x1, y1 = int(kp1['x']), int(kp1['y'])
            x2, y2 = int(kp2['x']), int(kp2['y'])

            cv2.circle(out_img, (x1, y1), 4, (0, 255, 0), 1)
            cv2.circle(out_img, (x2 + cols1, y2), 4, (0, 255, 0), 1)

            random_color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.line(out_img, (x1, y1), (x2 + cols1, y2), random_color, thickness=3)
        
        return out_img