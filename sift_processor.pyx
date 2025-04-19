# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, sqrt, atan2, cos, sin, fabs, M_PI as pi
from cython.parallel import prange

# Define Keypoint as a cdef class for better performance
cdef class Keypoint:
    cdef public double x, y, value
    cdef public int octave, scale
    
    def __init__(self, double x, double y, int octave, int scale, double value):
        self.x = x
        self.y = y
        self.octave = octave
        self.scale = scale
        self.value = value

def calculate_gradient(cnp.ndarray[cnp.float32_t, ndim=2] image):
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef cnp.ndarray[cnp.float32_t, ndim=2] Gx = np.empty((height, width), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] Gy = np.empty((height, width), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] magnitude = np.empty((height, width), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] orientation = np.empty((height, width), dtype=np.float32)
    
    cdef int i, j
    cdef float dx, dy
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            dx = image[i, j+1] - image[i, j-1]
            dy = image[i-1, j] - image[i+1, j]
            Gx[i,j] = dx
            Gy[i,j] = dy
            magnitude[i,j] = sqrt(dx*dx + dy*dy)
            orientation[i,j] = atan2(dy, dx) * 180 / pi
    
    return magnitude, orientation

def _find_extrema_in_octave(cnp.ndarray[cnp.float32_t, ndim=2] current_DOG,
                           cnp.ndarray[cnp.float32_t, ndim=2] above_DOG,
                           cnp.ndarray[cnp.float32_t, ndim=2] below_DOG,
                           int octave, int scale, float contrast_threshold=0.04):
    cdef int height = current_DOG.shape[0]
    cdef int width = current_DOG.shape[1]
    cdef list keypoints = []
    cdef float current_val
    cdef float neighbor_val
    cdef int i, j, di, dj, ds
    cdef bint is_extremum
    cdef float dxx, dyy, dxy, tr, det
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            current_val = current_DOG[i,j]
            if fabs(current_val) < contrast_threshold:
                continue
                
            # Check 26 neighbors
            is_extremum = True
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for ds in range(-1, 2):
                        if di == 0 and dj == 0 and ds == 0:
                            continue
                            
                        if ds == -1:
                            neighbor_val = below_DOG[i+di,j+dj]
                        elif ds == 1:
                            neighbor_val = above_DOG[i+di,j+dj]
                        else:
                            neighbor_val = current_DOG[i+di,j+dj]
                            
                        if (current_val > 0 and current_val <= neighbor_val) or \
                           (current_val < 0 and current_val >= neighbor_val):
                            is_extremum = False
                            break
                    if not is_extremum:
                        break
                if not is_extremum:
                    break
                    
            if is_extremum:
                # Edge rejection
                dxx = current_DOG[i,j+1] + current_DOG[i,j-1] - 2*current_val
                dyy = current_DOG[i+1,j] + current_DOG[i-1,j] - 2*current_val
                dxy = (current_DOG[i+1,j+1] + current_DOG[i-1,j-1] - 
                       current_DOG[i+1,j-1] - current_DOG[i-1,j+1]) / 4
                tr = dxx + dyy
                det = dxx * dyy - dxy * dxy
                
                if det > 0 and (tr * tr) / det < 14.5:  # (10+1)^2/10 = 12.1
                    keypoints.append(Keypoint(
                        x=j * (1 << octave),
                        y=i * (1 << octave),
                        octave=octave,
                        scale=scale,
                        value=current_val
                    ))
    
    return keypoints