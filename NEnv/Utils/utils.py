import numpy as np
from scipy.special import rel_entr
from torch.utils import data

from nsf import data as data_


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
