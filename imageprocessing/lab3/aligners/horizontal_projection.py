import numpy as np
from skimage.filters import sobel, threshold_local
from skimage.transform import rotate

class HorizontalProjectionAligner:
    """Digitalized document aligner based on horizontal projection."""

    def preprocess(self, image):
        borders = sobel(image)

        threshold = threshold_local(borders, 99)
        binary_image = borders > threshold

        return binary_image

    def fitness(self, rotated):
        profile = rotated.sum(axis=1)
        return np.square(profile - np.roll(profile, 1)).sum()

    def find_rotation_angle(self, preprocessed):
        fitnesses = {}

        for angle in range(-90, 90):
            rotated = rotate(preprocessed, angle)
            fitnesses[angle] = self.fitness(rotated)

        return max(fitnesses.items(), key=lambda item: item[1])[0]

    def align(self, image):
        preprocessed = self.preprocess(image)
        rotation_angle = self.find_rotation_angle(preprocessed)

        return rotate(image, rotation_angle, resize=True, preserve_range=True).astype(np.uint8)