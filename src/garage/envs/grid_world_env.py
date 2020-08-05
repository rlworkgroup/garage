"""Simple 2D Grid environment."""
import akro
import numpy as np

from garage import Environment, EnvSpec, EnvStep, StepType

MAPS = {
    'chain': ['GFFFFFFFFFFFFFSFFFFFFFFFFFFFG'],
    '4x4_safe': [
        'SFFF',
        'FWFW',
        'FFFW',
        'WFFG'
    ],
    '4x4': [
        'SFFF',
        'FHFH',
        'FFFH',
        'HFFG'
    ],
    '8x8': [
        'SFFFFFFF',
        'FFFFFFFF',
        'FFFHFFFF',
        'FFFFFHFF',
        'FFFHFFFF',
        'FHHFFFHF',
        'FHFFHFHF',
        'FFFHFFFG'
    ],
}   # yapf: disable


class GridWorldEnv(Environment):
    """A simply 2D grid environment.

    | 'S' : starting point
    | 'F' or '.': free space
    | 'W' or 'x': wall
    | 'H' or 'o': hole (terminates episode)
    | 'G' : goal
    """

    def __init__(self, desc='4x4', max_episode_length=None):
        """Initialize the environment.

        Args:
            desc (str): grid configuration key.
            max_episode_length (int): The maximum steps allowed for an episode.
        """
        if isinstance(desc, str):
            desc = MAPS[desc]
        desc = np.array(list(map(list, desc)))
        desc[desc == '.'] = 'F'
        desc[desc == 'o'] = 'H'
        desc[desc == 'x'] = 'W'
        self._desc = desc
        self._n_row, self._n_col = desc.shape
        (start_x, ), (start_y, ) = np.nonzero(desc == 'S')
        self._start_state = start_x * self._n_col + start_y
        self._state = None
        self._domain_fig = None

        self._step_cnt = None
        self._max_episode_length = max_episode_length

        self._action_space = akro.Discrete(4)
        self._observation_space = akro.Discrete(self._n_row * self._n_col)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=max_episode_length)

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return []

    def reset(self):
        """Resets the environment.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)

        """
        self._state = self._start_state
        self._step_cnt = 0

        return self._state, dict()

    def step(self, action):
        """Steps the environment.

        action map:
        0: left
        1: down
        2: right
        3: up

        Args:
            action (int): an int encoding the action

        Returns:
            EnvStep: The environment step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment has been
                constructed and `reset()` has not been called.
            NotImplementedError: if a next step in self._desc does not match
                known state type.
        """
        if self._step_cnt is None:
            raise RuntimeError('reset() must be called before step()!')

        possible_next_states = self._get_possible_next_states(
            self._state, action)

        probs = [x[1] for x in possible_next_states]
        next_state_idx = np.random.choice(len(probs), p=probs)
        next_state = possible_next_states[next_state_idx][0]

        next_x = next_state // self._n_col
        next_y = next_state % self._n_col

        next_state_type = self._desc[next_x, next_y]
        if next_state_type == 'H':
            done = True
            reward = 0.0
        elif next_state_type in ['F', 'S']:
            done = False
            reward = 0.0
        elif next_state_type == 'G':
            done = True
            reward = 1.0
        else:
            raise NotImplementedError

        self._state = next_state

        self._step_cnt += 1
        step_type = StepType.get_step_type(
            step_cnt=self._step_cnt,
            max_episode_length=self._max_episode_length,
            done=done)

        if step_type in (StepType.TERMINAL, StepType.TIMEOUT):
            self._step_cnt = None

        return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=reward,
                       observation=next_state,
                       env_info={},
                       step_type=step_type)

    def render(self, mode):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `Environment.render_modes`.
        """

    def visualize(self):
        """Creates a visualization of the environment."""

    def close(self):
        """Close the env."""

    def _get_possible_next_states(self, state, action):
        """Return possible next states and their probabilities.

        Only next states with nonzero probabilities will be returned.

        Args:
            state (list): start state
            action (int): action

        Returns:
            list: a list of pairs (s', p(s'|s,a))
        """
        x = state // self._n_col
        y = state % self._n_col
        coords = np.array([x, y])

        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_coords = np.clip(coords + increments[action], [0, 0],
                              [self._n_row - 1, self._n_col - 1])
        next_state = next_coords[0] * self._n_col + next_coords[1]
        state_type = self._desc[x, y]
        next_state_type = self._desc[next_coords[0], next_coords[1]]
        if next_state_type == 'W' or state_type == 'H' or state_type == 'G':
            return [(state, 1.)]
        else:
            return [(next_state, 1.)]
