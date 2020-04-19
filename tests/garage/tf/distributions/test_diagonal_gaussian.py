import numpy as np

from garage.tf.distributions import DiagonalGaussian


def test_kl():
    gaussian = DiagonalGaussian(dim=2)

    dist1 = dict(mean=np.array([0, 0]), log_std=np.array([0, 0]))
    dist2 = dict(mean=np.array([0, 0]), log_std=np.array([1, 1]))
    dist3 = dict(mean=np.array([1, 1]), log_std=np.array([0, 0]))

    assert np.isclose(gaussian.kl(dist1, dist1), 0)
    assert np.isclose(gaussian.kl(dist1, dist2),
                      2 * (1 + np.e**2) / (2 * np.e**2))
    assert np.isclose(gaussian.kl(dist3, dist1), 2 * 0.5)


def test_sample():
    gaussian = DiagonalGaussian(dim=2)
    dist = dict(mean=np.array([1, 1]), log_std=np.array([0, 0]))
    samples = [gaussian.sample(dist) for _ in range(10000)]
    assert np.isclose(np.mean(samples), 1, atol=0.1)
    assert np.isclose(np.var(samples), 1, atol=0.1)


def test_sample_sym():
    gaussian = DiagonalGaussian(dim=2)
    dist = dict(mean=np.array([1., 1.], dtype=np.float32),
                log_std=np.array([0., 0.], dtype=np.float32))
    samples = [gaussian.sample_sym(dist).numpy() for _ in range(10000)]
    assert np.isclose(np.mean(samples), 1, atol=0.1)
    assert np.isclose(np.var(samples), 1, atol=0.1)
