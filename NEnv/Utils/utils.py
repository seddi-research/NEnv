import numpy as np
from scipy.special import rel_entr
from torch.utils import data

import sys
from os import path

# directory reach
directory = path.path(__file__).abspath()

# Allowing imports for parent classes
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)

from nsf import data as data_
from nsf import utils



def luminance(R, G, B):
    return (0.2126 * R + 0.7152 * G + 0.0722 * B)


def luminance_rgb(RGB):
    return luminance(RGB[0], RGB[1], RGB[2])


def clamp(n, smallest, largest): return max(smallest, min(n, largest))


def toVector(theta, phi):
    sinPhi = np.sin(phi)
    cosPhi = np.cos(phi)

    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)

    return [sinTheta * sinPhi, cosTheta, sinTheta * cosPhi]


def toSpherical(direction):
    phi = np.arctan2(direction[0], direction[2])
    if phi < 0.:
        phi = phi + 2.0 * np.pi

    return np.arccos(direction[1]), phi


def KL(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon

    # divergence = np.mean(P*np.log(P/Q))
    return sum(rel_entr(P, Q))


def get_pdf_environment_map(envmap, number_of_points=500):
    num_points_per_axis = number_of_points
    bounds = np.array([
        [1e-3, 1 - 1e-3],
        [1e-3, 1 - 1e-3]
    ])

    grid_dataset = data_.TestGridDataset(
        num_points_per_axis=num_points_per_axis,
        bounds=bounds
    )
    grid_loader = data.DataLoader(
        dataset=grid_dataset,
        batch_size=1,
        drop_last=False
    )

    density_np = []
    for batch in grid_loader:
        for element in batch:
            pdf = envmap.m_distribution.PDF(element.cpu().numpy())
            density_np = np.concatenate(
                (density_np, [pdf])
            )

    probabilities = density_np.reshape(grid_dataset.X.shape)
    probabilities_gt = np.flip(probabilities, 1)
    return probabilities_gt


def get_gt_image(envmap, number_of_points=512):
    num_points_per_axis = number_of_points
    bounds = np.array([
        [1e-3, 1 - 1e-3],
        [1e-3, 1 - 1e-3]
    ])

    grid_dataset = data_.TestGridDataset(
        num_points_per_axis=num_points_per_axis,
        bounds=bounds
    )
    grid_loader = data.DataLoader(
        dataset=grid_dataset,
        batch_size=1,
        drop_last=False
    )

    rgbs_np_gt = []

    for batch in grid_loader:
        for element in batch:
            rgb = envmap.image[
                int(element.cpu().numpy()[0] * envmap.width), int(element.cpu().numpy()[1] * envmap.height)]
            rgbs_np_gt.append([rgb])

    rgbs_np_gt = np.array(rgbs_np_gt)
    rgbs_np_gt = rgbs_np_gt.reshape([grid_dataset.X.shape[0], grid_dataset.X.shape[1], 3])
    return rgbs_np_gt


def get_predicted_pdf(flow, device, number_of_points = 500):
    num_points_per_axis = number_of_points
    bounds = np.array([
        [1e-3, 1 - 1e-3],
        [1e-3, 1 - 1e-3]
    ])
    grid_dataset = data_.TestGridDataset(
        num_points_per_axis=num_points_per_axis,
        bounds=bounds
    )
    grid_loader = data.DataLoader(
        dataset=grid_dataset,
        batch_size=10000,
        drop_last=False
    )

    flow.eval()
    log_density_np = []
    for batch in grid_loader:
        batch = batch.to(device)
        log_density = flow.log_prob(batch)
        log_density_np = np.concatenate(
            (log_density_np, utils.tensor2numpy(log_density))
        )
    probabilities_pred = np.exp(log_density_np).reshape(grid_dataset.X.shape)
    probabilities_pred = probabilities_pred.T
    return probabilities_pred

def get_predicted_image(model, device, number_of_points = 512):
    num_points_per_axis = number_of_points
    bounds = np.array([
        [1e-3, 1 - 1e-3],
        [1e-3, 1 - 1e-3]
    ])
    grid_dataset = data_.TestGridDataset(
        num_points_per_axis=num_points_per_axis,
        bounds=bounds
    )
    grid_loader = data.DataLoader(
        dataset=grid_dataset,
        batch_size=50000,
        drop_last=False
    )

    model.eval()

    rgbs_np = []

    for batch in grid_loader:
        batch = batch.to(device)
        rgbs = model(batch)
        rgbs = utils.tensor2numpy(rgbs)
        for rgb in range(rgbs.shape[0]):
            rgbs_np.append(rgbs[rgb])

    rgbs_np = np.array(rgbs_np)
    rgbs_np = rgbs_np.reshape([grid_dataset.X.shape[0], grid_dataset.X.shape[1], 3])
    rgbs_np = rgbs_np.T
    return rgbs_np.transpose((1, -1, 0))

def get_predicted_image_rectangular(model, device, number_of_points = (1920, 1080)):
    num_points_per_axis = number_of_points
    bounds = np.array([
        [1e-3, 1 - 1e-3],
        [1e-3, 1 - 1e-3]
    ])
    grid_dataset = data_.TestGridDatasetRectangular(
        num_points_per_axis_x=num_points_per_axis[0],
        num_points_per_axis_y=num_points_per_axis[1],
        bounds=bounds
    )
    grid_loader = data.DataLoader(
        dataset=grid_dataset,
        batch_size=100000,
        drop_last=False
    )

    model.eval()

    rgbs_np = []

    for batch in grid_loader:
        batch = batch.to(device).half()
        rgbs = model(batch)
        rgbs = utils.tensor2numpy(rgbs)
        for rgb in range(rgbs.shape[0]):
            rgbs_np.append(rgbs[rgb])

    rgbs_np = np.array(rgbs_np)
    rgbs_np = rgbs_np.reshape([grid_dataset.X.shape[0], grid_dataset.X.shape[1], 3])
    rgbs_np = rgbs_np.T
    return rgbs_np.transpose((1, -1, 0))

def get_pdf_environment_map_rectangular(envmap, number_of_points = (2000,1000)):
    num_points_per_axis = number_of_points
    bounds = np.array([
        [1e-5, 1 - 1e-5],
        [1e-5, 1 - 1e-5]
    ])

    grid_dataset = data_.TestGridDatasetRectangular(
        num_points_per_axis_x=num_points_per_axis[0],
        num_points_per_axis_y=num_points_per_axis[1],
        bounds=bounds
    )

    grid_loader = data.DataLoader(
        dataset=grid_dataset,
        batch_size=10000,
        drop_last=False
    )

    density_np = []
    for batch in grid_loader:
        for element in batch:
            pdf = envmap.m_distribution.PDF(element.cpu().numpy())
            density_np = np.concatenate(
                (density_np, [pdf])
            )

    probabilities = density_np.reshape(grid_dataset.X.shape)
    probabilities_gt = np.flip(probabilities, 1)
    return probabilities_gt

def sample_rgb(envmap, N, proportional=False):
    """

    :param envmap: Environment map from which to sample
    :param N: number of samples
    :param proportional: set to true if you want the samples to follow the pdf of the envmap. Otherwise they are sampled uniformely at random.
    :return: samples in a numpy array
    """
    assert isinstance(N, int)
    uniform_samples =np.random.uniform(low=0, high=1, size=(N,2))
    samples = []
    rgbs = []

    if proportional:
        for s in range(N):
            uniform_sample = uniform_samples[s]
            envmap_sample0, envmap_sample1, rgb = envmap.SampleSphericalRGB(uniform_sample)
            x = (envmap_sample1 / np.pi * 0.5)
            y = (envmap_sample0 / np.pi)
            samples.append((y, x))
            rgbs.append(rgb)
    else:
        for s in range(N):
            uniform_sample = uniform_samples[s]
            phi = uniform_sample[0] * 2 * np.pi
            theta = uniform_sample[1] * np.pi
            x = uniform_sample[0]
            y = uniform_sample[1]
            samples.append((y, x))
            rgb = envmap.EvalAngles(theta, phi)
            rgbs.append(rgb)

    return np.array(samples), np.array(rgbs)

def get_predicted_pdf_rectangular(flow, device, number_of_points = 500):
    num_points_per_axis = number_of_points
    bounds = np.array([
        [1e-5, 1 - 1e-5],
        [1e-5, 1 - 1e-5]
    ])
    grid_dataset = data_.TestGridDatasetRectangular(
        num_points_per_axis_x=num_points_per_axis[0],
        num_points_per_axis_y=num_points_per_axis[1],
        bounds=bounds
    )
    grid_loader = data.DataLoader(
        dataset=grid_dataset,
        batch_size=50000,
        drop_last=False
    )

    flow.eval()
    log_density_np = []
    for batch in grid_loader:
        batch = batch.to(device)
        log_density = flow.log_prob(batch)
        log_density_np = np.concatenate(
            (log_density_np, utils.tensor2numpy(log_density))
        )
    probabilities_pred = np.exp(log_density_np).reshape(grid_dataset.X.shape)
    probabilities_pred = probabilities_pred.T
    return probabilities_pred