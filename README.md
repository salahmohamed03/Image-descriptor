# Image-descriptor
## Overview

The application features a PyQt-based UI with two main functionalities: Corner Detection (using Harris and Lambda methods) and Feature Matching (using SIFT with SSD and NCC methods) to process and compare images.

## UI

Interactive interface with tabs for Corner Detection and Feature Matching, supporting real-time processing and visualization.

## Corner Detection

Techniques to identify key corner points in images.

| Method            | Image                                      | Description                                      |
|-------------------|--------------------------------------------|--------------------------------------------------|
| Harris-Corner-Detection | ![Harris-Corner-Detection](https://github.com/user-attachments/assets/e5508ce6-f9d2-432b-8910-6830e1a73ef2) | Uses Sobel gradients, Gaussian smoothing, and Harris response with thresholding to detect corners. |
| Lambda-Corner-Detection | ![Lambda-Corner-Detection](https://github.com/user-attachments/assets/4f329067-aba7-4ea0-b717-6acd1bbaf81a) | Computes minimum eigenvalue of the structure tensor, with thresholding for strong corners. |

## SIFT

Scale-Invariant Feature Transform for keypoint detection and description.

| Method            | Image                                      | Description                                      |
|-------------------|--------------------------------------------|--------------------------------------------------|
| SIFT-Keypoint-Detection | ![SIFT-Keypoint-Detection](https://github.com/user-attachments/assets/58a2e215-3ec5-4e8f-bb3f-738af8db7cb4) | Builds Gaussian and DoG pyramids, detects extrema, refines keypoints, and generates 128D descriptors. |

## Feature Matching

Methods to match keypoints between two images using SIFT descriptors.

| Method            | Image                                      | Description                                      |
|-------------------|--------------------------------------------|--------------------------------------------------|
| SSD-Matching      | ![SSD-Matching](https://github.com/user-attachments/assets/3f837364-d209-49fa-a28e-18d1e9780a7b) | Uses Sum of Squared Differences with a 0.8 ratio test to find reliable matches. |
| NCC-Matching      | ![NCC-Matching](https://github.com/user-attachments/assets/5b0a9241-421c-44a2-b12f-d9e4c77d22fa) | Applies Normalized Cross-Correlation with a 0.8 ratio test, handling scale and lighting differences. |
| Draw-Matches      | ![Draw-Matches](https://github.com/user-attachments/assets/f566d2ac-cc30-4494-a5df-456684509ef6) | Combines images, draws green circles and colored lines for matches, supporting grayscale and color visuals. |

## Getting Started

1. Clone the repository: `git clone https://github.com/Ayatullah-ahmed/Team-7-Computer-Vision-Assignment-3.git`
2. Install dependencies (e.g., PyQt, NumPy, OpenCV).
3. Run the UI to explore the techniques.

## Contributors

<table>
  <tr>
        <td align="center">
      <a href="https://github.com/salahmohamed03">
        <img src="https://avatars.githubusercontent.com/u/93553073?v=4" width="250px;" alt="Salah Mohamed"/>
        <br />
        <sub><b>Salah Mohamed</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Ayatullah-ahmed" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/125223938?v=" width="250px;" alt="Ayatullah Ahmed"/>
        <br />
        <sub><b>Ayatullah Ahmed</b></sub>
      </a>
    </td>
        <td align="center">
      <a href="https://github.com/Abdelrahman0Sayed">
        <img src="https://avatars.githubusercontent.com/u/113141265?v=4" width="250px;" alt="Abdelrahman Sayed"/>
        <br />
        <sub><b>Abdelrahman Sayed</b></sub>
      </a>
    </td>
        </td>
        <td align="center">
      <a href="https://github.com/AhmeedRaafatt">
        <img src="https://avatars.githubusercontent.com/u/125607744?v=4" width="250px;" alt="Ahmed Raffat"/>
        <br />
        <sub><b>Ahmed Rafaat</b></sub>
      </a>
    </td>
  </tr>
</table>
