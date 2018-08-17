import numpy as np

from garage.core.serializable import Serializable
from garage.envs.mujoco.sawyer.sawyer_env import Configuration
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnv
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnvWrapper
from garage.misc.overrides import overrides
from gym.envs.robotics import rotations


class PushEnv(SawyerEnv):
    def __init__(self,
                 direction="up",
                 easy_gripper_init=True,
                 randomize_start_pos=False,
                 control_method='task_space_control',
                 **kwargs):
        def start_goal_config():
            # center = self.sim.data.get_geom_xpos('target2')
            if randomize_start_pos:
                xy = [
                    np.random.uniform(0.6, 0.8),
                    np.random.uniform(-0.35, 0.35)
                ]
            else:
                xy = [0.62, 0.17]
            d = 0.15
            delta = np.array({
                "up": (d, 0),
                "down": (-d, 0),
                "left": (0, d),
                "right": (0, -d)
            }[direction])
            if easy_gripper_init:
                # position gripper besides the block
                gripper_pos = np.concatenate([xy - delta, [0.07]])
            else:
                # position gripper above the block
                gripper_pos = np.concatenate([xy, [0.2]])
            if control_method == 'task_space_control':
                start = Configuration(
                    gripper_pos=gripper_pos,
                    gripper_state=1,
                    object_grasped=False,
                    object_pos=(0.8, 0, 0),
                    joint_pos=None)
                goal = Configuration(
                    gripper_pos=(0.5, 0, 0),
                    gripper_state=1,
                    object_grasped=False,
                    object_pos=(0.5, 0, 0),
                    joint_pos=None)
            else:
                if easy_gripper_init:
                    if direction != "up":
                        raise Warning("Easy gripper init has only been implemented for the up direction so far.")
                    jpos = np.array({
                        "up": [
                            0, -0.97, 0, 2., 0, 0.5, 1.65
                        ],
                        "down": [
                            -0.12526904, 0.29675812, 0.06034621, -0.55948609,
                            -0.03694355, 1.8277617, -1.54921871
                        ],
                        "left": [
                            -0.36766702, 0.62033507, 0.00376033, -1.33212273,
                            0.06092402, 2.29230268, -1.7248123
                        ],
                        "right": [
                            5.97299145e-03, 6.46604393e-01, 1.40055632e-03,
                            -1.22810430e+00, 9.04236294e-03, 2.13193649e+00,
                            -1.38572576e+00
                        ]
                    }[direction])
                else:
                    jpos = np.array([
                        0, -1.1, 0, 1.3, 0, 1.4, 1.65
                    ])
                start = Configuration(
                    gripper_pos=gripper_pos,
                    gripper_state=0,
                    object_grasped=False,
                    object_pos=np.concatenate([xy, [0.04]]),
                    joint_pos=jpos)
                goal = Configuration(
                    gripper_pos=None,
                    gripper_state=0,
                    object_grasped=False,
                    object_pos=np.concatenate([xy + delta, [0.04]]),
                    joint_pos=None)
            return start, goal

        def achieved_goal_fn(env: SawyerEnv):
            return env.object_position

        def desired_goal_fn(env: SawyerEnv):
            return env._goal_configuration.object_pos

        def reward_fn(env: SawyerEnv, achieved_goal, desired_goal, info: dict):
            gripper_pos = info["gripper_position"]
            object_pos = info["object_position"]

            upright_gripper = np.array([np.pi, 0, np.pi])
            gripper_rot = rotations.mat2euler(env.sim.data.get_site_xmat('grip'))
            gripper_norot = -np.linalg.norm(np.sin(upright_gripper) - np.sin(gripper_rot))

            if not easy_gripper_init and direction == "up":
                if "_traj_phase" not in env._episode_data or env._episode_data["_traj_phase"] == 0:
                    env._episode_data["_traj_phase"] = 0
                    # Gripper start on top of block -> has to move in front of it
                    if gripper_pos[2] < 0.05 and np.linalg.norm(gripper_pos - object_pos + np.array([0.1, 0, -0.05])) < 0.2 and abs(gripper_norot) < 0.5:
                        env._episode_data["_traj_phase"] += 1
                        print("Reached phase", env._episode_data["_traj_phase"])
                    return -gripper_pos[2] - 3. * np.linalg.norm(gripper_pos - object_pos + np.array([0.1, 0, -0.05])) + gripper_pos[1] - 2. * np.linalg.norm(gripper_pos[1] - object_pos[1]) + 2. * gripper_norot

            reach_block = -np.linalg.norm(gripper_pos - object_pos)
            block_goal = -np.linalg.norm(achieved_goal - desired_goal)
            block_down = -object_pos[2]

            dt = env.sim.nsubsteps * env.sim.model.opt.timestep
            block_norot = -np.linalg.norm(env.sim.data.get_geom_xvelr('object0') * dt)

            return reach_block + 3. * block_goal + .4 * block_down + 0.3 * block_norot + .1 * gripper_norot

        super(PushEnv, self).__init__(
            start_goal_config=start_goal_config,
            achieved_goal_fn=achieved_goal_fn,
            desired_goal_fn=desired_goal_fn,
            file_path="push.xml" if control_method == "task_space_control" else "push_poscontrol.xml",
            collision_whitelist=[],
            collision_penalty=0.,
            control_method=control_method,
            reward_fn=reward_fn,
            send_done_signal=False,
            **kwargs)
        self._easy_gripper_init = easy_gripper_init

    def get_obs(self):
        gripper_pos = self.gripper_position
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        object_pos = self.object_position
        object_velp = self.sim.data.get_geom_xvelp('object0') * dt
        object_velp -= grip_velp
        grasped = self.has_object
        if self._control_method == "position_control":
            obs = np.concatenate([self.joint_positions, self.object_position, self.gripper_position])
        else:
            # obs = np.concatenate([gripper_pos, object_pos])
            # obs = np.concatenate([gripper_pos])
            obs = np.array([0.])

        achieved_goal = self._achieved_goal_fn(self)
        desired_goal = self._desired_goal_fn(self)

        # achieved_goal_qpos = np.concatenate((achieved_goal, [1, 0, 0, 0]))
        # self.sim.data.set_joint_qpos('achieved_goal:joint', achieved_goal_qpos)
        # desired_goal_qpos = np.concatenate((desired_goal, [1, 0, 0, 0]))
        # self.sim.data.set_joint_qpos('desired_goal:joint', desired_goal_qpos)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
            'gripper_state': self.gripper_state,
            'gripper_pos': gripper_pos.copy(),
            'has_object': grasped,
            'object_pos': object_pos.copy()
        }

    # @overrides
    # @property
    # def action_space(self):
    #     if self._control_method == 'torque_control':
    #         return super(SawyerEnv, self).action_space
    #     elif self._control_method == 'task_space_control':
    #         # TODO revert to xyz actions
    #         return Box(
    #             np.array([-0.15, -0.15]),
    #             np.array([0.15, 0.15]),
    #             dtype=np.float32)
    #     elif self._control_method == 'position_control':
    #         return Box(
    #             low=np.full(9, -0.04), high=np.full(9, 0.04), dtype=np.float32)
    #     else:
    #         raise NotImplementedError

    # @overrides
    # def step(self, action):
    #     assert action.shape == self.action_space.shape
    #
    #     # Note: you MUST copy the action if you modify it
    #     a = action.copy()
    #
    #     # Clip to action space
    #     a *= self._action_scale
    #     a = np.clip(a, self.action_space.low, self.action_space.high)
    #
    #     if self._control_method == "torque_control":
    #         self.forward_dynamics(a)
    #         self.sim.forward()
    #     elif self._control_method == "task_space_control":
    #         reset_mocap2body_xpos(self.sim)
    #         self.sim.data.mocap_pos[0, :2] = self.sim.data.mocap_pos[0, :2] + a[:2]
    #         self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
    #         # self.set_gripper_state(a[3])
    #         for _ in range(5):
    #             self.sim.step()
    #         self.sim.forward()
    #     elif self._control_method == "position_control":
    #         curr_pos = self.joint_positions
    #
    #         next_pos = np.clip(
    #             a + curr_pos,
    #             self.joint_position_space.low,
    #             self.joint_position_space.high
    #         )
    #         # self.sim.data.ctrl[:] = next_pos
    #         for _ in range(1):
    #             self.sim.data.ctrl[:] = next_pos
    #             self.sim.step()
    #             self.sim.forward()
    #     else:
    #         raise NotImplementedError
    #     self._step += 1
    #
    #     obs = self.get_obs()
    #     self._achieved_goal = obs.get('achieved_goal')
    #     self._desired_goal = self._goal_configuration.gripper_pos
    #
    #     # collision checking is expensive so cache the value
    #     in_collision = self.in_collision
    #
    #     info = {
    #         "l": self._step,
    #         "grasped": obs["has_object"],
    #         "gripper_state": obs["gripper_state"],
    #         "gripper_position": obs["gripper_pos"],
    #         "object_position": obs["object_pos"],
    #         "is_success": self._is_success,
    #         "in_collision": in_collision,
    #     }
    #
    #     self._achieved_goal = self.object_position
    #     self._desired_goal = self._goal_configuration.object_pos
    #
    #     r = self.compute_reward(
    #         achieved_goal=self._achieved_goal,
    #         desired_goal=self._desired_goal,
    #         info=info)
    #
    #     self._is_success = self._success_fn(self, self._achieved_goal,
    #                                         self._desired_goal, info)
    #     done = False
    #
    #     # control cost
    #     r -= self._control_cost_coeff * np.linalg.norm(a)
    #
    #     # collision detection
    #     if in_collision:
    #         r -= self._collision_penalty
    #         if self._terminate_on_collision:
    #             done = True
    #
    #     if self._is_success:
    #         r = self._completion_bonus
    #         # done = True
    #
    #     info["r"] = r
    #     info["d"] = done
    #
    #     return obs, r, done, info


class SimplePushEnv(SawyerEnvWrapper, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.reward_range = None
        self.metadata = None
        super().__init__(PushEnv(*args, **kwargs))
