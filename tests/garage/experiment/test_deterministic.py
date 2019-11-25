"""Tests for deterministic.py"""
import random

import numpy as np
import tensorflow as tf
import torch

from garage.experiment import deterministic


def test_deterministic_pytorch():
    """Test deterministic behavior of PyTorch"""
    deterministic.set_seed(111)
    rand_tensor = torch.rand((5, 5))
    deterministic_tensor = torch.Tensor(
        [[0.715565920, 0.913992643, 0.281857729, 0.258099794, 0.631108642],
         [0.600053012, 0.931192935, 0.215290189, 0.603278518, 0.732785344],
         [0.185717106, 0.510067403, 0.754451334, 0.288391531, 0.577469587],
         [0.035843492, 0.102626860, 0.341910362, 0.439984798, 0.634111166],
         [0.622391582, 0.633447766, 0.857972443, 0.157199264, 0.785320759]])

    assert torch.all(torch.eq(rand_tensor, deterministic_tensor))


def test_deterministic_tensorflow():
    """Test deterministic behavior of Tensorflow"""
    deterministic.set_seed(77)
    rand_tensor = tf.random_uniform((5, 5))
    with tf.Session() as sess:
        rand_tensor = sess.run(rand_tensor)
    deterministic_tensor = np.array(
        [[0.11519885, 0.41889858, 0.93573880, 0.64490880, 0.76444733],
         [0.36086679, 0.46140290, 0.35727130, 0.46127295, 0.82892287],
         [0.50611700, 0.060229897, 0.85028017, 0.37228084, 0.49440527],
         [0.60247135, 0.27734910, 0.24438739, 0.58618486, 0.92419887],
         [0.35655558, 0.32704484, 0.93260970, 0.40895236, 0.86852560]],
        dtype=np.float32)

    assert np.all(np.array_equal(rand_tensor, deterministic_tensor))


def test_deterministic_numpy():
    """Test deterministic behavior of numpy"""
    deterministic.set_seed(22)
    rand_tensor = np.random.rand(5, 5)
    deterministic_tensor = np.array(
        [[0.20846054, 0.48168106, 0.42053804, 0.859182, 0.17116155],
         [0.33886396, 0.27053283, 0.69104135, 0.22040452, 0.81195092],
         [0.01052687, 0.5612037, 0.81372619, 0.7451003, 0.18911136],
         [0.00614087, 0.77204387, 0.95783217, 0.70193788, 0.29757827],
         [0.76799274, 0.68821832, 0.38718348, 0.61520583, 0.42755524]])
    assert np.allclose(rand_tensor, deterministic_tensor)


def test_deterministic_random():
    """Test deterministic behavior of random"""
    deterministic.set_seed(55)
    rand_array = [random.random() for _ in range(10)]
    deterministic_array = [
        0.09033985426934954, 0.9506335645634441, 0.14997105299598545,
        0.7393703706762795, 0.8412423959349363, 0.7471369518620469,
        0.30193759566924927, 0.35162393686161975, 0.7218626135761532,
        0.9656464075038401
    ]

    assert rand_array == deterministic_array
