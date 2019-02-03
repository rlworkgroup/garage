""" Wrapper for the dm_control viewer which allows single-stepping """
import dm_control.viewer.application as dm_viewer_app
import glfw


class DmControlViewer(dm_viewer_app.Application):
    def render(self):
        # Don't try to render into closed windows
        if not self._window._context:
            return

        self._render_once()

        # Just keep rendering if we're paused, but hold onto control flow
        while self._pause_subject.value:
            self._render_once()

    def _render_once(self):
        # See https://github.com/deepmind/dm_control/blob/92f9913013face0468442cd0964d5973ea2089ea/dm_control/viewer/gui/glfw_gui.py#L280  # noqa: E501
        window = self._window
        tick_func = self._tick_func
        if (window._context
                and not glfw.window_should_close(window._context.window)):
            pixels = tick_func()
            with window._context.make_current() as ctx:
                ctx.call(window._update_gui_on_render_thread,
                         window._context.window, pixels)
            window._mouse.process_events()
            window._keyboard.process_events()
        else:
            window.close()

    def launch(self, environment_loader, policy=None):
        # See https://github.com/deepmind/dm_control/blob/92f9913013face0468442cd0964d5973ea2089ea/dm_control/viewer/application.py#L304  # noqa: E501
        if environment_loader is None:
            raise ValueError('"environment_loader" argument is required.')
        if callable(environment_loader):
            self._environment_loader = environment_loader
        else:
            self._environment_loader = lambda: environment_loader
        self._policy = policy
        self._load_environment(zoom_to_scene=True)

        def tick():
            self._viewport.set_size(*self._window.shape)
            self._tick()
            return self._renderer.pixels

        self._tick_func = tick

        # Start unpaused
        self._pause_subject.value = False

    def close(self):
        self._window.close()
