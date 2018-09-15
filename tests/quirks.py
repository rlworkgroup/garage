"""Documented breakages and quirks caused by dependencies."""

# openai/gym environments known to not implement render()
#
# e.g.
# > gym/core.py", line 111, in render
# >     raise NotImplementedError
# > NotImplementedError
#
# Tests calling render() on these should verify they raise NotImplementedError
# ```
# with self.assertRaises(NotImplementedError):
#     env.render()
# ```
KNOWN_GYM_RENDER_NOT_IMPLEMENTED = [
    # Please keep alphabetized
    "Blackjack-v0",
    "GuessingGame-v0",
    "HotterColder-v0",
    "NChain-v0",
    "Roulette-v0",
]

# openai/gym environments known to have a broken close() function.
#
# e.g.
# > gym/envs/robotics/robot_env.py", line 86, in close
# >     self.viewer.finish()
# > AttributeError: 'MjViewer' object has no attribute 'finish'
# >
#
# Tests calling close() on these should verify they raise AttributeError
# ```
# with self.assertRaises(NotImplementedError):
#     env.close()
# ```
#
# TODO: file a bug at openai/gym about this
KNOWN_GYM_CLOSE_BROKEN = [
    # Please keep alphabetized
    "Ant-v2",
    "FetchPickAndPlace-v1",
    "FetchPickAndPlaceDense-v1",
    "FetchPush-v1",
    "FetchPushDense-v1",
    "FetchReach-v1",
    "FetchReachDense-v1",
    "FetchSlide-v1",
    "FetchSlideDense-v1",
    "HalfCheetah-v2",
    "HandManipulateBlock-v0",
    "HandManipulateBlockDense-v0",
    "HandManipulateBlockFull-v0",
    "HandManipulateBlockFullDense-v0",
    "HandManipulateBlockRotateParallel-v0",
    "HandManipulateBlockRotateParallelDense-v0",
    "HandManipulateBlockRotateXYZ-v0",
    "HandManipulateBlockRotateXYZDense-v0",
    "HandManipulateBlockRotateZ-v0",
    "HandManipulateBlockRotateZDense-v0",
    "HandManipulateEgg-v0",
    "HandManipulateEggDense-v0",
    "HandManipulateEggFull-v0",
    "HandManipulateEggFullDense-v0",
    "HandManipulateEggRotate-v0",
    "HandManipulateEggRotateDense-v0",
    "HandManipulatePen-v0",
    "HandManipulatePenDense-v0",
    "HandManipulatePenFull-v0",
    "HandManipulatePenFullDense-v0",
    "HandManipulatePenRotate-v0",
    "HandManipulatePenRotateDense-v0",
    "HandReach-v0",
    "HandReachDense-v0",
    "Hopper-v2",
    "Humanoid-v2",
    "HumanoidStandup-v2",
    "InvertedDoublePendulum-v2",
    "InvertedPendulum-v2",
    "Pusher-v2",
    "Reacher-v2",
    "Striker-v2",
    "Swimmer-v2",
    "Thrower-v2",
    "Walker2d-v2",
]

KNOWN_GYM_NOT_CLOSE_VIEWER = [
    # Please keep alphabetized
    "gym.envs.mujoco",
    "gym.envs.robotics"
]
