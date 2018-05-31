import ctypes
from ctypes import byref
from threading import Lock

from mujoco_py import const as C
from mujoco_py import functions
from mujoco_py import MjRenderContext
from mujoco_py import MjSim
from mujoco_py.builder import cymj
from mujoco_py.generated.const import CAT_ALL
import mujoco_py
import glfw
# https://github.com/openai/mujoco-py/blob/6ac6ac203a875ef35b1505827264cadccbfd9f05/mujoco_py/builder.py#L61

PyMjrRect = cymj.PyMjrRect
PyMjvCamera = cymj.PyMjvCamera


class EmbeddedViewer(object):
    def __init__(self):
        self.last_render_time = 0
        self.cam = PyMjvCamera()

        self.running = False
        self.speedtype = 1
        self.window = None
        self.model = None
        self.gui_lock = Lock()

        self.last_button = 0
        self.last_click_time = 0
        self.button_left_pressed = False
        self.button_middle_pressed = False
        self.button_right_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.frames = []

    def set_model(self, model):
        self.model = model
        if model:
            self.sim = MjSim(self.model)
            self.data = self.sim.data
            self.con = MjRenderContext(self.sim)
            self.scn = self.con.scn
            self.vopt = self.con.vopt
        else:
            self.data = None
        if self.running:
            if model:
                functions.mjr_makeContext(self.model, self.con, 150)
            else:
                functions.mjr_makeContext(None, self.con, 150)
            self.render()
        if model:
            self.autoscale()

    def autoscale(self):
        self.cam.lookat[0] = self.sim.stat.center[0]
        self.cam.lookat[1] = self.sim.stat.center[1]
        self.cam.lookat[2] = self.sim.stat.center[2]
        self.cam.distance = 1.0 * self.sim.stat.extent
        self.cam.camid = -1
        self.cam.trackbodyid = -1
        if self.window:
            width, height = glfw.get_framebuffer_size(self.window)
            functions.mjv_moveCamera(self.model, None, width, height, self.scn,
                                     self.cam)

    def get_rect(self):
        rect = PyMjrRect()
        rect.width, rect.height = glfw.get_framebuffer_size(self.window)
        return rect

    def record_frame(self, **kwargs):
        self.frames.append({'pos': self.sim.data.qpos, 'extra': kwargs})

    def clear_frames(self):
        self.frames = []

    def render(self):
        rect = self.get_rect()
        width = rect.width, height = rect.height

        functions.mjv_addGeoms(self.model, self.data, self.vopt, None, CAT_ALL,
                               self.scn)
        functions.mjv_updateCamera(self.model, self.data, self.scn)
        functions.mjv_moveCamera(self.model, None, width / height,
                                 height / height, self.scn, self.cam)
        functions.mjr_render(rect, self.scn, self.con)

    def render_internal(self):
        if not self.data:
            return
        self.gui_lock.acquire()
        self.render()

        self.gui_lock.release()

    def start(self, window):
        self.running = True

        width, height = glfw.get_framebuffer_size(window)
        width1, height = glfw.get_window_size(window)
        self.scale = width * 1.0 / width1

        self.window = window

        functions.mjv_makeScene(self.scn, 1000)

        if self.model:
            functions.mjr_makeContext(self.model, self.con, 150)
            self.autoscale()
        else:
            functions.mjr_makeContext(None, self.con, 150)

    def handle_mouse_move(self, window, xpos, ypos):

        # no buttons down: nothing to do
        if not self.button_left_pressed \
                and not self.button_middle_pressed \
                and not self.button_right_pressed:
            return

        # compute mouse displacement, save
        dx = int(self.scale * xpos) - self.last_mouse_x
        dy = int(self.scale * ypos) - self.last_mouse_y
        self.last_mouse_x = int(self.scale * xpos)
        self.last_mouse_y = int(self.scale * ypos)

        # require model
        if not self.model:
            return

        # get current window size
        width, height = glfw.get_framebuffer_size(self.window)

        # get shift key state
        mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS \
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

        # determine action based on mouse button
        action = None
        if self.button_right_pressed:
            action = C.MOUSE_MOVE_H if mod_shift else C.MOUSE_MOVE_V
        elif self.button_left_pressed:
            action = C.MOUSE_ROTATE_H if mod_shift else C.MOUSE_ROTATE_V
        else:
            action = C.MOUSE_ZOOM

        self.gui_lock.acquire()

        functions.mjv_moveCamera(self.model, action, dx / height, dy / height,
                                 self.scn, self.cam)

        self.gui_lock.release()

    def handle_mouse_button(self, window, button, act, mods):
        # update button state
        self.button_left_pressed = \
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_middle_pressed = \
            glfw.get_mouse_button(
                window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.button_right_pressed = \
            glfw.get_mouse_button(
                window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

        # update mouse position
        x, y = glfw.get_cursor_pos(window)
        self.last_mouse_x = int(self.scale * x)
        self.last_mouse_y = int(self.scale * y)

        if not self.model:
            return

        self.gui_lock.acquire()

        # save info
        if act == glfw.PRESS:
            self.last_button = button
            self.last_click_time = glfw.get_time()

        self.gui_lock.release()

    def handle_scroll(self, window, x_offset, y_offset):
        # require model
        if not self.model:
            return

        # get current window size
        width, height = glfw.get_framebuffer_size(window)

        # scroll
        self.gui_lock.acquire()
        functions.mjv_moveCamera(self.model, C.MOUSE_ZOOM, 0,
                                 (-20 * y_offset) / height, self.scn, self.cam)

        self.gui_lock.release()

    def should_stop(self):
        return glfw.window_should_close(self.window)

    def loop_once(self):
        self.render()
        # Swap front and back buffers
        glfw.swap_buffers(self.window)
        # Poll for and process events
        glfw.poll_events()

    def finish(self):
        glfw.terminate()
        functions.mjr_freeContext(self.con)
        functions.mjv_freeScene(self.scn)
        self.running = False
