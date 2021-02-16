# facial_landmark_and_image_morphing
This repository contains Python codes to detect facial landmarks of people and some animals. After finding these points, face morphing is applied to images by using delaunay triangulation.  
## Installation  
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required libraries.  
```bash
pip install dlib opencv-python numpy moviepy
```
I used Python 3.8.5 and cmake should be in system PATH.  
## Usage  
```bash
python landmark_points.py
```  
or  
```bash
python face_morphing_w_triangulation.py
```  
## Explanations  
### Faces with Landmark Points  
![Alt Text](https://github.com/emreslyn/facial_landmark_and_image_morphing/blob/main/outputs/part1_with_landmarks_collage.png)  
### Detected Face  
![Alt Text](https://github.com/emreslyn/facial_landmark_and_image_morphing/blob/main/outputs/part1_with_rectangle_kimbodnia.png)  
### Image Morphing  Example  
![Alt Text](https://github.com/emreslyn/facial_landmark_and_image_morphing/blob/main/outputs/transformation.gif)  
