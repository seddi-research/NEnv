import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from NEnv.Utils.distributions import Distribution2D
from NEnv.Utils.utils import toVector, toSpherical, luminance_rgb

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class Envmap:
    def __init__(self, fileName, gamma=1, resolution=(4000, 200)):
        """
        Class that encodes and is able to sample from a GT environment map
        @param fileName: Path to HDR environment map
        @type fileName: str
        @param gamma: Gamma correction to apply to the environment map
        @type gamma: float
        @param resolution: Target resolution onto which to resample the original envmap
        @type resolution: tuple
        """

        self.image = cv2.imread(fileName, flags=cv2.IMREAD_ANYDEPTH)[:, :, :3]

        self.image = cv2.resize(self.image, resolution, interpolation=cv2.INTER_AREA)

        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        self.image = cv2.flip(self.image, 1)
        self.image = cv2.cvtColor(np.float32(self.image), cv2.COLOR_BGR2RGB) ** gamma
        self.imagePlot = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)

        wData = []
        for v in range(0, self.height):
            sinTheta = np.sin(np.pi * (float(v) + .5) / float(self.height))
            for u in range(0, self.width):
                wData.append(luminance_rgb(self.image[u, v]) * sinTheta)

        # Use data to compute the 2D distribution
        self.m_distribution = Distribution2D(wData, self.width, self.height)

    def plot(self):
        plt.imshow(self.imagePlot, interpolation='nearest', aspect='auto')

    def sample_direction(self, sample):
        uv, pdf = self.m_distribution.Sample(sample);
        theta = uv[1] * np.pi
        # Reverse phi
        phi = (1.0 - uv[0]) * 2.0 * np.pi
        direction = toVector(theta, phi)
        sinTheta = np.sin(theta)
        pdf = pdf / (2. * np.pi * np.pi * sinTheta)
        if sinTheta == 0.0:
            pdf = 0.0

        return self.image[int(uv[0] * self.width), int(uv[1] * self.height)], direction, pdf

    def sample_spherical_direction(self, sample):
        uv, pdf = self.m_distribution.Sample(sample);

        theta = uv[1] * np.pi

        # Reverse phi
        phi = (1.0 - uv[0]) * 2.0 * np.pi

        sinTheta = np.sin(theta)
        pdf = pdf / (2. * np.pi * np.pi * sinTheta)
        if sinTheta == 0.0:
            pdf = 0.0

        return self.image[int(uv[0] * self.width), int(uv[1] * self.height)], theta, phi, pdf

    def sample_spherical_simple(self, sample):
        uv = self.m_distribution.SampleValues(sample);

        theta = uv[1] * np.pi

        # Reverse phi
        phi = (1.0 - uv[0]) * 2.0 * np.pi

        return theta, phi

    def pdf(self, theta, phi):

        sinTheta = np.sin(theta)
        if sinTheta == 0.0:
            return 0.0
        else:
            return self.m_distribution.PDF([phi * 0.5 / np.pi, theta / np.pi]) / (2. * np.pi * np.pi * sinTheta)

    def pdf_angles(self, direction):
        theta, phi = toSpherical([-direction[0], direction[1], direction[2]])
        return self.pdf(theta, phi)

    def eval_angles(self, theta, phi):
        return self.image[int(phi * .5 / np.pi * self.width), int(theta / np.pi * self.height)]

    def eval(self, direction):
        theta, phi = toSpherical([-direction[0], direction[1], direction[2]])
        return self.eval_angles(theta, phi)
