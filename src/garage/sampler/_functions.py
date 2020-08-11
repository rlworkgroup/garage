"""Functions used by multiple Samplers or Workers."""
from garage import Environment
from garage.sampler.env_update import EnvUpdate


def _apply_env_update(old_env, env_update):
    """Use any non-None env_update as a new environment.

    A simple env update function. If env_update is not None, it should be
    the complete new environment.

    This allows changing environments by passing the new environment as
    `env_update` into `obtain_samples`.

    Args:
        old_env (Environment): Environment to updated.
        env_update (Environment or EnvUpdate or None): The environment to
            replace the existing env with. Note that other implementations
            of `Worker` may take different types for this parameter.

    Returns:
        Environment: The updated environment (may be a different object from
            `old_env`).
        bool: True if an update happened.

    Raises:
        TypeError: If env_update is not one of the documented types.

    """
    if env_update is not None:
        if isinstance(env_update, EnvUpdate):
            return env_update(old_env), True
        elif isinstance(env_update, Environment):
            if old_env is not None:
                old_env.close()
            return env_update, True
        else:
            raise TypeError('Unknown environment update type.')
    else:
        return old_env, False
