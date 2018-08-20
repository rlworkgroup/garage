import math
import os.path as osp
import tempfile
import xml.etree.ElementTree as ET

import glfw  # noqa: I100
import gym
from mujoco_py import functions
from mujoco_py import load_model_from_path
from mujoco_py import MjRenderContext
from mujoco_py import MjSim
from mujoco_py import MjViewer
# https://github.com/openai/mujoco-py/blob/6ac6ac203a875ef35b1505827264cadccbfd9f05/mujoco_py/builder.py#L61
from mujoco_py.builder import cymj
from mujoco_py.generated.const import CAT_ALL
import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.mujoco.gather.embedded_viewer import EmbeddedViewer
from garage.envs.mujoco.mujoco_env import BIG
from garage.envs.mujoco.mujoco_env import MODEL_DIR
from garage.envs.util import flat_dim
from garage.misc import autoargs
from garage.misc import logger
from garage.misc.overrides import overrides

APPLE = 0
BOMB = 1
PyMjrRect = cymj.PyMjrRect


class GatherViewer(MjViewer):
    def __init__(self, env):
        self.env = env
        super(GatherViewer, self).__init__(self.env.env.sim)
        green_ball_model = load_model_from_path(
            osp.abspath(osp.join(MODEL_DIR, 'green_ball.xml')))
        self.green_ball_renderer = EmbeddedViewer()
        self.green_ball_model = green_ball_model
        self.green_ball_sim = MjSim(self.green_ball_model)
        self.green_ball_renderer.set_model(green_ball_model)
        red_ball_model = load_model_from_path(
            osp.abspath(osp.join(MODEL_DIR, 'red_ball.xml')))
        self.red_ball_renderer = EmbeddedViewer()
        self.red_ball_model = red_ball_model
        self.red_ball_sim = MjSim(self.red_ball_model)
        self.red_ball_renderer.set_model(red_ball_model)

    def start(self):
        self.green_ball_renderer.start(self.window)
        self.red_ball_renderer.start(self.window)

    def handle_mouse_move(self, window, xpos, ypos):
        self.green_ball_renderer.handle_mouse_move(window, xpos, ypos)
        self.red_ball_renderer.handle_mouse_move(window, xpos, ypos)

    def handle_scroll(self, window, x_offset, y_offset):
        self.green_ball_renderer.handle_scroll(window, x_offset, y_offset)
        self.red_ball_renderer.handle_scroll(window, x_offset, y_offset)

    def get_rect(self):
        rect = PyMjrRect()
        rect.width, rect.height = glfw.get_framebuffer_size(self.window)
        return rect

    def render(self):
        super(GatherViewer, self).render()
        ctx = MjRenderContext(self.env.env.sim)
        scn = ctx.scn
        con = ctx.con
        functions.mjv_makeScene(scn, 1000)
        scn.camera[0].frustum_near = 0.05
        scn.camera[1].frustum_near = 0.05

        for obj in self.env.objects:
            x, y, typ = obj
            qpos = np.zeros_like(self.green_ball_sim.data.qpos)
            qpos[0] = x
            qpos[1] = y
            if typ == APPLE:
                self.green_ball_sim.data.qpos[:] = qpos
                self.green_ball_sim.forward()
                self.green_ball_renderer.render()
                self.green_ball_ctx = MjRenderContext(self.green_ball_sim)
                functions.mjv_addGeoms(self.green_ball_sim.model,
                                       self.green_ball_sim.data,
                                       self.green_ball_ctx.vopt,
                                       self.green_ball_ctx.pert, CAT_ALL, scn)
            else:
                self.red_ball_sim.data.qpos[:] = qpos
                self.red_ball_sim.forward()
                self.red_ball_renderer.render()
                self.red_ball_ctx = MjRenderContext(self.red_ball_sim)
                functions.mjv_addGeoms(self.red_ball_sim.model,
                                       self.red_ball_sim.data,
                                       self.red_ball_ctx.vopt,
                                       self.red_ball_ctx.pert, CAT_ALL, scn)

        functions.mjv_addGeoms(self.env.env.sim.model, self.env.env.sim.data,
                               ctx.vopt, ctx.pert, CAT_ALL, scn)
        functions.mjr_render(self.green_ball_renderer.get_rect(), scn, con)

        try:
            import OpenGL.GL as GL
        except ImportError:
            return

        def draw_rect(x, y, width, height):
            # start drawing a rectangle
            GL.glBegin(GL.GL_QUADS)
            # bottom left point
            GL.glVertex2f(x, y)
            # bottom right point
            GL.glVertex2f(x + width, y)
            # top right point
            GL.glVertex2f(x + width, y + height)
            # top left point
            GL.glVertex2f(x, y + height)

        def refresh2d(width, height):
            GL.glViewport(0, 0, width, height)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GL.glOrtho(0.0, width, 0.0, height, 0.0, 1.0)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()

        GL.glBegin(GL.GL_QUADS)
        GL.glLoadIdentity()
        width, height = glfw.get_framebuffer_size(self.window)
        refresh2d(width, height)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_BLEND)

        GL.glColor4f(0.0, 0.0, 0.0, 0.8)
        draw_rect(10, 10, 300, 100)

        apple_readings, bomb_readings = self.env.get_readings()
        for idx, reading in enumerate(apple_readings):
            if reading > 0:
                GL.glColor4f(0.0, 1.0, 0.0, reading)
                draw_rect(20 * (idx + 1), 10, 5, 50)
        for idx, reading in enumerate(bomb_readings):
            if reading > 0:
                GL.glColor4f(1.0, 0.0, 0.0, reading)
                draw_rect(20 * (idx + 1), 60, 5, 50)


class GatherEnv(gym.Wrapper, Serializable):
    MODEL_CLASS = None
    ORI_IND = None

    @autoargs.arg(
        'n_apples', type=int, help='Number of apples in each episode')
    @autoargs.arg('n_bombs', type=int, help='Number of bombs in each episode')
    @autoargs.arg(
        'activity_range',
        type=float,
        help='The span for generating objects '
        '(x, y in [-range, range])')
    @autoargs.arg(
        'robot_object_spacing',
        type=float,
        help='Number of objects in each episode')
    @autoargs.arg(
        'catch_range',
        type=float,
        help='Minimum distance range to catch an object')
    @autoargs.arg(
        'n_bins', type=float, help='Number of objects in each episode')
    @autoargs.arg(
        'sensor_range',
        type=float,
        help='Maximum sensor range (how far it can go)')
    @autoargs.arg(
        'sensor_span',
        type=float,
        help='Maximum sensor span (how wide it can span), in '
        'radians')
    def __init__(self,
                 n_apples=8,
                 n_bombs=8,
                 activity_range=6.,
                 robot_object_spacing=2.,
                 catch_range=1.,
                 n_bins=10,
                 sensor_range=6.,
                 sensor_span=math.pi,
                 coef_inner_rew=0.,
                 dying_cost=-10,
                 *args,
                 **kwargs):
        self.n_apples = n_apples
        self.n_bombs = n_bombs
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.catch_range = catch_range
        self.n_bins = n_bins
        self.sensor_range = sensor_range
        self.sensor_span = sensor_span
        self.coef_inner_rew = coef_inner_rew
        self.dying_cost = dying_cost
        self.objects = []
        self.render_width = 512
        self.render_height = 512
        model_cls = self.__class__.MODEL_CLASS
        if not model_cls:
            raise NotImplementedError("MODEL_CLASS unspecified!")
        xml_path = osp.join(MODEL_DIR, model_cls.FILE)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")
        attrs = dict(
            type="box", conaffinity="1", rgba="0.8 0.9 0.8 1", condim="3")
        walldist = self.activity_range + 1
        ET.SubElement(
            worldbody, "geom",
            dict(
                attrs,
                name="wall1",
                pos="0 -%d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom",
            dict(
                attrs,
                name="wall2",
                pos="0 %d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom",
            dict(
                attrs,
                name="wall3",
                pos="-%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom",
            dict(
                attrs,
                name="wall4",
                pos="%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        _, file_path = tempfile.mkstemp(suffix=".xml", text=True)
        tree.write(file_path)
        # pylint: disable=not-callable
        inner_env = model_cls(*args, file_path=file_path, **kwargs)
        # pylint: enable=not-callable
        super().__init__(inner_env)

        # Redefine observation space
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        self.observation_space = gym.spaces.Box(ub * -1, ub, dtype=np.float32)

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    def reset(self, also_wrapped=True):
        self.objects = []
        existing = set()
        while len(self.objects) < self.n_apples:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x**2 + y**2 < self.robot_object_spacing**2:
                continue
            if (x, y) in existing:
                continue
            typ = APPLE
            self.objects.append((x, y, typ))
            existing.add((x, y))
        while len(self.objects) < self.n_apples + self.n_bombs:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x**2 + y**2 < self.robot_object_spacing**2:
                continue
            if (x, y) in existing:
                continue
            typ = BOMB
            self.objects.append((x, y, typ))
            existing.add((x, y))

        if also_wrapped:
            self.env.reset()
        return self.get_current_obs()

    def step(self, action):
        _, inner_rew, done, info = self.env.step(action)
        info['inner_rew'] = inner_rew
        info['outer_rew'] = 0
        if done:
            return Step(self.get_current_obs(), self.dying_cost, done,
                        **info)  # give a -10 rew if the robot dies
        com = self.env.get_body_com("torso")
        x, y = com[:2]
        reward = self.coef_inner_rew * inner_rew
        new_objs = []
        for obj in self.objects:
            ox, oy, typ = obj
            # object within zone!
            if (ox - x)**2 + (oy - y)**2 < self.catch_range**2:
                if typ == APPLE:
                    reward = reward + 1
                    info['outer_rew'] = 1
                else:
                    reward = reward - 1
                    info['outer_rew'] = -1
            else:
                new_objs.append(obj)
        self.objects = new_objs
        done = len(self.objects) == 0
        return Step(self.get_current_obs(), reward, done, **info)

    def get_readings(
            self):  # equivalent to get_current_maze_obs in maze_env.py
        # compute sensor readings
        # first, obtain current orientation
        apple_readings = np.zeros(self.n_bins)
        bomb_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.env.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects,
            key=lambda o: (o[0] - robot_x)**2 + (o[1] - robot_y)**2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins

        ori = self.get_ori()  # overwrite this for Ant!

        for ox, oy, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y)**2 + (ox - robot_x)**2)**0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                raise ValueError("Variable angle is not a valid number")
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            intensity = 1.0 - dist / self.sensor_range
            if typ == APPLE:
                apple_readings[bin_number] = intensity
            else:
                bomb_readings[bin_number] = intensity
        return apple_readings, bomb_readings

    def get_current_robot_obs(self):
        return self.env.get_current_obs()

    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.env.get_current_obs()
        apple_readings, bomb_readings = self.get_readings()
        return np.concatenate([self_obs, apple_readings, bomb_readings])

    # space of only the robot observations (they go first in the get current
    # obs)
    @property
    def robot_observation_space(self):
        shp = self.get_current_robot_obs().shape
        ub = BIG * np.ones(shp)
        return gym.spaces.Box(ub * -1, ub, dtype=np.float32)

    @property
    def maze_observation_space(self):
        shp = np.concatenate(self.get_readings()).shape
        ub = BIG * np.ones(shp)
        return gym.spaces.Box(ub * -1, ub, dtype=np.float32)

    @property
    def action_bounds(self):
        return self.env.action_bounds

    @property
    def viewer(self):
        return self.get_viewer()

    def action_from_key(self, key):
        return self.env.action_from_key(key)

    def get_viewer(self):
        if self.env.viewer is None:
            self.env.viewer = GatherViewer(self)
            self.env.viewer.start()
            r = self.env.viewer.get_rect()
            self.render_width, self.render_height = r.width, r.height
        return self.env.viewer

    def stop_viewer(self):
        if self.env.viewer:
            self.env.viewer.finish()

    def render(self, mode='human', close=False):  # pylint: disable=R1710
        # The render function returns immediately since it's not working
        # properly. This has to be addressed for
        # https://github.com/rlworkgroup/garage/issues/323.
        return
        if mode == 'rgb_array':
            viewer = self.get_viewer()
            self.env.render()
            img = viewer.read_pixels(
                width=self.render_width,
                height=self.render_height,
                depth=False)
            img = img[::-1, :, :]
            # transpose image s.t. img.shape[0] = width, img.shape[1] = height
            img = np.swapaxes(img, 0, 1)
            return img
        elif mode == 'human':
            self.get_viewer()
            self.env.render()
        if close:
            self.stop_viewer()

    def get_ori(self):
        """
        First it tries to use a get_ori from the wrapped env. If not
        successful, falls back to the default based on the ORI_IND specified in
        Maze (not accurate for quaternions)
        """
        obj = self.env
        while not hasattr(obj, 'get_ori') and hasattr(obj, 'env'):
            obj = obj.env
        try:
            return obj.get_ori()
        except (NotImplementedError, AttributeError):
            pass
        return self.env.sim.data.qpos[self.__class__.ORI_IND]

    @overrides
    def log_diagnostics(self, paths, log_prefix='Gather', *args, **kwargs):
        # we call here any logging related to the gather, strip the maze obs
        # and call log_diag with the stripped paths we need to log the purely
        # gather reward!!
        with logger.tabular_prefix(log_prefix + '_'):
            gather_undiscounted_returns = [
                sum(path['env_infos']['outer_rew']) for path in paths
            ]
            logger.record_tabular_misc_stat(
                'Return', gather_undiscounted_returns, placement='front')
        stripped_paths = []
        for path in paths:
            stripped_path = {}
            for k, v in path.items():
                stripped_path[k] = v
            stripped_path['observations'] = \
                stripped_path['observations'][
                    :, :flat_dim(self.env.observation_space)]
            #  this breaks if the obs of the robot are d>1 dimensional (not a
            #  vector)
            stripped_paths.append(stripped_path)
        with logger.tabular_prefix('wrapped_'):
            if 'env_infos' in paths[0].keys(
            ) and 'inner_rew' in paths[0]['env_infos'].keys():
                wrapped_undiscounted_return = np.mean(
                    [np.sum(path['env_infos']['inner_rew']) for path in paths])
                logger.record_tabular('AverageReturn',
                                      wrapped_undiscounted_return)
            self.env.log_diagnostics(
                stripped_paths
            )  # see swimmer_env.py for a scketch of the maze plotting!
