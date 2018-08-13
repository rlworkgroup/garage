import numpy as np

from gym.spaces import Box
from gym.envs.robotics import rotations
from gym.envs.robotics.utils import reset_mocap_welds, reset_mocap2body_xpos

from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnv, Configuration
from garage.misc.overrides import overrides


class PushEnv(SawyerEnv):
    def __init__(self, direction="up", easy_gripper_init=True, **kwargs):
        def start_goal_config():
            # center = self.sim.data.get_geom_xpos('target2')
            xy = [np.random.uniform(0.6, 0.8), np.random.uniform(-0.35, 0.35)]
            d = 0.15
            delta = np.array({
                "up":    ( d,  0),
                "down":  (-d,  0),
                "left":  ( 0,  d),
                "right": ( 0, -d)
            }[direction])
            if easy_gripper_init:
                # position gripper besides the block
                gripper_pos = np.concatenate([xy - delta, [0.07]])
            else:
                # position gripper above the block
                gripper_pos = np.concatenate([xy, [0.2]])
            start = Configuration(
                gripper_pos=gripper_pos,
                gripper_state=0,
                object_grasped=False,
                object_pos=np.concatenate([xy, [0.03]]))
            goal = Configuration(
                gripper_pos=None,
                gripper_state=0,
                object_grasped=False,
                object_pos=np.concatenate([xy + delta, [0.03]]))
            return start, goal

        def achieved_goal_fn(env: SawyerEnv):
            return env.object_position

        def desired_goal_fn(env: SawyerEnv):
            return env._goal_configuration.object_pos

        super(PushEnv, self).__init__(
            start_goal_config=start_goal_config,
            achieved_goal_fn=achieved_goal_fn,
            desired_goal_fn=desired_goal_fn,
            file_path="push.xml",
            **kwargs)

    def get_obs(self):
        gripper_pos = self.gripper_position
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip') * dt

        object_pos = self.object_position
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velp -= grip_velp
        grasped = self.has_object
        obs = np.concatenate([gripper_pos, object_pos])

        achieved_goal = self._achieved_goal_fn(self)
        desired_goal = self._desired_goal_fn(self)

        achieved_goal_qpos = np.concatenate((achieved_goal, [1, 0, 0, 0]))
        self.sim.data.set_joint_qpos('achieved_goal:joint', achieved_goal_qpos)
        desired_goal_qpos = np.concatenate((desired_goal, [1, 0, 0, 0]))
        self.sim.data.set_joint_qpos('desired_goal:joint', desired_goal_qpos)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
            'gripper_state': self.gripper_state,
            'gripper_pos': gripper_pos.copy(),
            'has_object': grasped,
            'object_pos': object_pos.copy()
        }

    @overrides
    @property
    def action_space(self):
        if self._control_method == 'torque_control':
            return super(SawyerEnv, self).action_space
        elif self._control_method == 'task_space_control':
            return Box(
                np.array([-0.15, -0.15, -0.15]),
                np.array([0.15, 0.15, 0.15]),
                dtype=np.float32)
        elif self._control_method == 'position_control':
            return Box(
                low=np.full(7, -0.04), high=np.full(7, 0.04), dtype=np.float32)
        else:
            raise NotImplementedError

    @overrides
    def step(self, action):
        # Clip to action space
        assert action.shape == self.action_space.shape
        a = action.copy()  # Note: you MUST copy the action if you modify it
        a *= self._action_scale
        a = np.clip(a, self.action_space.low, self.action_space.high)

        if self._control_method == "torque_control":
            self.forward_dynamics(a)
            self.sim.forward()
        elif self._control_method == "task_space_control":
            reset_mocap2body_xpos(self.sim)
            self.sim.data.mocap_pos[0, :3] = self.sim.data.mocap_pos[0, :3] + a[:3]
            self.sim.data.mocap_quat[:] = np.array([0, 1, 0, 0])
            # self.set_gripper_state(a[3])
            self.sim.forward()
            for _ in range(1):
                self.sim.step()
        elif self._control_method == "position_control":
            low, high = self.joint_position_limits
            curr_pos = self.joint_positions
            next_pos = np.clip(a + curr_pos, low, high)
            for i in range(7):
                self.sim.data.set_joint_qpos("right_j{}".format(i),
                                             next_pos[i])
            for _ in range(3):
                self.sim.forward()
            self.sim.step()

            # Verify the execution of the action.
            # for i in range(7):
            #     curr_pos = self.sim.data.get_joint_qpos('right_j{}'.format(i))
            #     d = np.absolute(curr_pos - next_pos[i])
            #     assert d < 1e-2, \
            #         "Joint right_j{} failed to reach the desired qpos.\nError: {}\t Desired: {}\t Current: {}" \
            #         .format(i, d, next_pos[i], curr_pos)
        else:
            raise NotImplementedError
        self._step += 1

        obs = self.get_obs()
        self._achieved_goal = obs.get('achieved_goal')
        self._desired_goal = obs.get('desired_goal')

        info = {
            "l": self._step,
            "grasped": obs["has_object"],
            "gripper_state": obs["gripper_state"],
            "gripper_position": obs["gripper_pos"],
            "object_position": obs["object_pos"],
            "is_success": self._is_success
        }

        r = self.compute_reward(
            achieved_goal=obs.get('achieved_goal'),
            desired_goal=obs.get('desired_goal'),
            info=info) + 0.02 - np.linalg.norm(self.gripper_position - self.object_position)

        self._is_success = self._success_fn(self, self._achieved_goal,
                                            self._desired_goal, info)
        done = False
        if self._is_success:
            # r = self._completion_bonus
            done = True

        info["r"] = r
        info["d"] = done

        # control cost
        r -= self._control_cost_coeff * np.linalg.norm(a)

        return obs, r, done, info
