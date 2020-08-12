"""Base class of HalfCheetah meta-environments."""
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
import numpy as np

from garage import EnvSpec


class HalfCheetahEnvMetaBase(HalfCheetahEnv_):
    """Base class of HalfCheetah meta-environments.

    Code is adapted from
    https://github.com/tristandeleu/pytorch-maml-rl/blob/493e677e724aa67a531250b0e215c8dbc9a7364a/maml_rl/envs/mujoco/half_cheetah.py

    Which was in turn adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py

    Args:
        task (dict): Subclass specific task information.

    """

    def __init__(self, task):
        self._task = task
        super().__init__()
        self.spec = EnvSpec(action_space=self.action_space,
                            observation_space=self.observation_space)

    def _get_obs(self):
        """Get a low-dimensional observation of the state.

        Returns:
            np.ndarray: Contains the flattened angle quaternion, angular
                velocity quaternion, and cartesian position.

        """
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com('torso').flat,
        ]).astype(np.float32).flatten()

    def viewer_setup(self):
        """Start the viewer."""
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        # This code was inheritted, so we'll ignore this access violation for
        # now.
        # pylint: disable=protected-access
        self.viewer._hide_overlay = True

    def __getstate__(self):
        """See `Object.__getstate__.

        Returns:
            dict: The instance’s dictionary to be pickled.

        """
        return dict(task=self._task)

    def __setstate__(self, state):
        """See `Object.__setstate__.

        Args:
            state (dict): Unpickled state of this object.

        """
        self.__init__(task=state['task'])
