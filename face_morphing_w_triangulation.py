import numpy as np
from utils import get_two_triangles, write_videos

image, image2, img1_triangles, img2_triangles = get_two_triangles("images\\aydemirakbas.png","images\\cat.jpg","aydemirakbas.png","cat.png",np.load("images\\cat_landmarks.npy"))
write_videos(img1_triangles,img2_triangles,image,image2,"outputs\\part3_aydemirToCat.mp4")
image, image2, img1_triangles, img2_triangles = get_two_triangles("images\\deniro.jpg","images\\gorilla.jpg","deniro.png","gorilla.png",np.load("images\\gorilla_landmarks.npy"))
write_videos(img1_triangles,img2_triangles,image,image2,"outputs\\part3_deniroToGorilla.mp4")
image, image2, img1_triangles, img2_triangles = get_two_triangles("images\\kimbodnia.png","images\\panda.jpg","kimbodnia.png","panda.png",np.load("images\\panda_landmarks.npy"))
write_videos(img1_triangles,img2_triangles,image,image2,"outputs\\part3_kimbodniaToPanda.mp4")