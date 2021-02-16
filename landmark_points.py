import cv2
import numpy as np
from utils import draw_rectangle,paint_landmark_points

location = "outputs\\part1_with_rectangle_"
image = cv2.imread("images\\aydemirakbas.png")
draw_rectangle(image,location+"aydemirakbas.png")
image = cv2.imread("images\\deniro.jpg")
draw_rectangle(image,location+"deniro.png")
image = cv2.imread("images\\kimbodnia.png")
draw_rectangle(image,location+"kimbodnia.png")

location = "outputs\\part1_with_landmarks_collage.png"
image = cv2.imread("images\\aydemirakbas.png")
image1 = paint_landmark_points(image)
image = cv2.imread("images\\deniro.jpg")
image2 = paint_landmark_points(image)
image = cv2.imread("images\\kimbodnia.png")
image3 = paint_landmark_points(image)

image = cv2.imread("images\\cat.jpg")
image4 = paint_landmark_points(image,np.load("images\\cat_landmarks.npy"))
image = cv2.imread("images\\gorilla.jpg")
image5 = paint_landmark_points(image,np.load("images\\gorilla_landmarks.npy"))
image = cv2.imread("images\\panda.jpg")
image6 = paint_landmark_points(image,np.load("images\\panda_landmarks.npy"))

top_image = cv2.hconcat([image6, image4, image5]) #Concatenate animal images horizontally
bottom_image = cv2.hconcat([image2, image1, image3]) #Concatenate human images horizontally
collage = cv2.vconcat([top_image, bottom_image]) # Concatenate top_image and bottom_image vertically
cv2.imwrite(location, collage) #Write image on specified place with "location"