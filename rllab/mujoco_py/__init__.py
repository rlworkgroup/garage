from rllab.mujoco_py.mjviewer import MjViewer
from rllab.mujoco_py.mjcore import MjModel
from rllab.mujoco_py.mjcore import register_license
import os
from rllab.mujoco_py.mjconstants import *

register_license(os.path.join(os.path.dirname(__file__),
                              '../../vendor/mujoco/mjkey.txt'))
