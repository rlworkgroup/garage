"""Example launcher file for experiment on simulation sawyer with trpo."""

import sys

import moveit_commander
import numpy as np
import rospy

from garage.baselines import LinearFeatureBaseline
from garage.contrib.ros.envs.sawyer import ReacherEnv
from garage.envs.util import spec
from garage.misc.instrument import run_experiment
from garage.theano.algos import TRPO
from garage.theano.policies import GaussianMLPPolicy

INITIAL_ROBOT_JOINT_POS = {
    'right_j0': -0.140923828125,
    'right_j1': -1.2789248046875,
    'right_j2': -3.043166015625,
    'right_j3': -2.139623046875,
    'right_j4': -0.047607421875,
    'right_j5': -0.7052822265625,
    'right_j6': -1.4102060546875,
}


def run_task(*_):
    """Run task function."""
    initial_goal = np.array([0.6, -0.1, 0.40])

    # Initialize moveit_commander
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node('trpo_sim_sawyer_reacher_exp', anonymous=True)

    env = ReacherEnv(
        initial_goal,
        initial_joint_pos=INITIAL_ROBOT_JOINT_POS,
        simulated=True)

    rospy.on_shutdown(env.shutdown)

    env.initialize()

    policy = GaussianMLPPolicy(env_spec=spec(env), hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=spec(env))

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=100,
        discount=0.99,
        step_size=0.01,
        plot=False,
        force_batch_sampler=True,
    )
    algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    plot=False,
)
