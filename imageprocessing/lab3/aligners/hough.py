import numpy as np
from scipy.stats import mode
from skimage.filters import sobel, threshold_local
from skimage.transform import hough_line, hough_line_peaks, rotate

class HoughAligner:
    """Digitalized document aligner based on Hough transform."""

    def preprocess(self, image):
        borders = sobel(image)

        threshold = threshold_local(borders, 99)
        binary_image = borders > threshold

        return binary_image

    def find_rotation_angle(self, preprocessed):
        hspace, angles, distances = hough_line(preprocessed)

        hspace, angles, distances = hough_line_peaks(hspace, angles, distances)

        rad_angle = mode(angles)[0]

        deg_angle = np.rad2deg(rad_angle)

        return deg_angle + 90 if deg_angle < 0 else deg_angle - 90

    def align(self, image):
        preprocessed = self.preprocess(image)
        rotation_angle = self.find_rotation_angle(preprocessed)

        return rotate(image, rotation_angle, resize=True, preserve_range=True).astype(np.uint8)
    