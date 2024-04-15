import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def get_equalization_transform_of_img(img):
    hist, bins = np.histogram(img,bins = 256,range = (0,256),density = True);
    u = np.zeros(hist.shape);
    y = np.zeros(hist.shape);
    u[0] = hist[0];
    y[0] = 0;
    for k in range(1, 256):
        u[k]=hist[k]+u[k-1];
        y[k]= round(((u[k]-u[0])/(1-u[0]))*(256-1));
    equalization_transform = y;
    
    return equalization_transform


def perform_global_hist_transform(img):
    equalized_img = np.zeros(img.shape);
    equalized_img_hist = get_equalization_transform_of_img(img); 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            equalized_img[i,j] = equalized_img_hist[img[i,j]];
    return equalized_img
