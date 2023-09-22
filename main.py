import numpy as np
import matplotlib.pyplot as plt

def grayscale(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-kernel_size//2)**2 +
                                                                            (y-kernel_size//2)**2)/(2*sigma**2)),
                                                                            (kernel_size, kernel_size))
    return kernel / np.sum(kernel)

def convolution2d(image, kernel):
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    height, width = image.shape
    padded_image = np.pad(image, pad_width=((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    result = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            result[y, x] = np.abs(np.sum(padded_image[y:y + kernel_size, x:x + kernel_size] * kernel))

    return result
def sobel_gradients(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_x = convolution2d(image, sobel_x)
    gradient_y = convolution2d(image, sobel_y)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_orientation = np.arctan2(gradient_y, gradient_x)
    return gradient_magnitude, gradient_orientation

def non_maximum_suppression(gradient_magnitude, gradient_orientation):
    height, width = gradient_magnitude.shape
    suppressed = np.zeros((height, width), dtype=np.float32)
    angle_quantized = (gradient_orientation * 180.0 / np.pi) % 180.0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            angle = angle_quantized[y, x]
            q, r = 0, 0
            if 0 <= angle < 22.5 or 157.5 <= angle < 180:
                q = gradient_magnitude[y, x + 1]
                r = gradient_magnitude[y, x - 1]
            elif 22.5 <= angle < 67.5:
                q = gradient_magnitude[y + 1, x - 1]
                r = gradient_magnitude[y - 1, x + 1]
            elif 67.5 <= angle < 112.5:
                q = gradient_magnitude[y + 1, x]
                r = gradient_magnitude[y - 1, x]
            elif 112.5 <= angle < 157.5:
                q = gradient_magnitude[y - 1, x - 1]
                r = gradient_magnitude[y + 1, x + 1]
            if gradient_magnitude[y, x] >= q and gradient_magnitude[y, x] >= r:
                suppressed[y, x] = gradient_magnitude[y, x]
    return suppressed

def double_thresholding(image, low_threshold, high_threshold):
    strong = 255
    weak = 50
    strong_edges = (image >= high_threshold)
    low_to_high_edges = (image >= low_threshold) & (image < high_threshold)
    image[strong_edges] = strong
    image[low_to_high_edges] = weak
    return image

def edge_tracking_hysteresis(image):
    weak = 50
    strong = 255
    height, width = image.shape
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if image[y, x] == weak:
                if np.max(image[y - 1:y + 2, x - 1:x + 2]) == strong:
                    image[y, x] = strong
                else:
                    image[y, x] = 0
    image[image != strong] = 0
    return image


def canny_edge_detector(image, kernel_size=5):
    low_threshold = 50
    high_threshold = 110
    gray = grayscale(image)
    kernel = gaussian_kernel(kernel_size, sigma=3)
    blurred_image = convolution2d(gray, kernel)
    # Resize the blurred image to the original size
    height, width = gray.shape
    blurred_image = blurred_image[:height, :width]
    gradient_magnitude, gradient_orientation = sobel_gradients(blurred_image)
    suppressed = non_maximum_suppression(gradient_magnitude, gradient_orientation)
    thresholded = double_thresholding(suppressed, low_threshold, high_threshold)
    edges = edge_tracking_hysteresis(thresholded)
    return blurred_image, gradient_magnitude, suppressed, thresholded, edges

# Load the image (assuming the path is correct)
input_image = plt.imread("//Users/nikunjpatel/Desktop/MTE 203/Project_2/Bridge-5.webp")

# Set the threshold values
low_threshold = 50
high_threshold = 110

# Apply Canny edge detection
blurred_image, gradient_magnitude, suppressed_image, thresholded_image, edges = canny_edge_detector(input_image)

# Display the images
plt.figure(figsize=(15, 15))
plt.subplot(231)
plt.imshow(input_image, cmap='gray')
plt.title("Input image")
plt.axis('off')


# plt.subplot(231)
# plt.imshow(blurred_image, cmap='gray')
# plt.title("Gaussian Blur")
# plt.axis('off')

# plt.subplot(231)
# plt.imshow(gradient_magnitude, cmap='gray')
# plt.title("Sobel Operator")
# plt.axis('off')
#
# plt.subplot(231)
# plt.imshow(suppressed_image, cmap='gray')
# plt.title("Non-Max Suppression")
# plt.axis('off')
# #
# DOUBLE THRESHOLDING
# plt.subplot(232)
# plt.imshow(thresholded_image, cmap='gray')
# plt.title("Double Thresholding")
# plt.axis('off')
# # #
# # THIS IS EDGE TRACKING BY HYSTERESIS
# plt.subplot(233)
# plt.imshow(edges, cmap='gray')
# plt.title("Final Output")
# plt.axis('off')

plt.tight_layout()
plt.show()
