"""EmbeddingSpec class."""


class EmbeddingSpec:
    """EmbeddingSpec class.

    Args:
        input_space (akro.Space): The input space of the env.
        latent_space (akro.Space): The latent space of the env.

    """

    def __init__(self, input_space, latent_space):
        self.input_space = input_space
        self.latent_space = latent_space
