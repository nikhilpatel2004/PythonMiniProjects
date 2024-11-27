import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
    """Read an image file and display it."""
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    return img

def edge_mask(img, line_size, blur_value):
    """Apply edge detection to an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blue = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blue, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

def color_quantization(img, k):
    """Apply color quantization to an image."""
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    return result.reshape(img.shape)

def cartoon(blurred, edges):
    """Create a cartoon-like image."""
    c = cv2.bitwise_and(blurred, blurred, mask=edges)
    plt.imshow(c)
    plt.title("Cartoonfied image")
    plt.show()

filename = "image.jpg"
img = read_file(filename)
org_img = np.copy(img)

line_size, blur_value = 5, 5
edges = edge_mask(img, line_size, blur_value)
plt.imshow(edges, cmap="binary")
plt.show()

img_quantiz = color_quantization(img, k=9)
plt.imshow(img_quantiz)
plt.show()

blurred = cv2.bilateralFilter(img, d=3, sigmaColor=200, sigmaSpace=200)
plt.imshow(blurred)
plt.show()

cartoon(blurred, edges)