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
    deterministic.set_seed(0)
    with tf.compat.v1.Session() as sess:
        rand_tensor = sess.run(
            tf.random.uniform((5, 5), seed=0, dtype=tf.dtypes.float32))
    deterministic_tensor = np.array(
        [[0.10086262, 0.9701668, 0.8487642, 0.04828131, 0.04852307],
         [0.77747464, 0.844468, 0.41707492, 0.5099584, 0.6552025],
         [0.9881507, 0.36698937, 0.37789786, 0.69118714, 0.99544394],
         [0.4662125, 0.9912039, 0.6973165, 0.7741407, 0.8881662],
         [0.03854167, 0.97539485, 0.23024535, 0.83840847, 0.79527795]],
        dtype=np.float32)

    assert np.allclose(rand_tensor, deterministic_tensor)


def test_deterministic_tfp_seed_stream():
    """Test deterministic behavior of TFP SeedStream"""
    deterministic.set_seed(0)
    with tf.compat.v1.Session() as sess:
        rand_tensor = sess.run(
            tf.random.uniform((5, 5),
                              seed=deterministic.get_tf_seed_stream(),
                              dtype=tf.dtypes.float32))
        sess.run(tf.random.uniform((5, 5), dtype=tf.dtypes.float32))
        rand_tensor2 = sess.run(
            tf.random.uniform((5, 5),
                              seed=deterministic.get_tf_seed_stream(),
                              dtype=tf.dtypes.float32))
    deterministic_tensor = np.array(
        [[0.10550332, 0.14218152, 0.5544759, 0.3720839, 0.6899766],
         [0.47086394, 0.5401237, 0.21653509, 0.42823565, 0.6927656],
         [0.16598761, 0.48356044, 0.36901915, 0.97140956, 0.07564807],
         [0.6694747, 0.21241283, 0.72315156, 0.631876, 0.34476352],
         [0.8718543, 0.4879316, 0.76272845, 0.04737151, 0.39661574]],
        dtype=np.float32)
    deterministic_tensor2 = np.array(
        [[0.9950017, 0.52794397, 0.7703887, 0.8688295, 0.78926384],
         [0.6301824, 0.45042813, 0.6257613, 0.7717335, 0.8412994],
         [0.30846167, 0.71520185, 0.13243473, 0.8455602, 0.01623428],
         [0.01353145, 0.23445582, 0.36002636, 0.3576231, 0.61981404],
         [0.47964382, 0.55043316, 0.3270856, 0.7003857, 0.53755534]],
        dtype=np.float32)

    assert np.allclose(rand_tensor, deterministic_tensor)
    assert np.allclose(rand_tensor2, deterministic_tensor2)


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
