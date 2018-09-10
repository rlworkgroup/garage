"""Double Deep Q Network.

An optimal policy is derived by selecting the highest-valued action in each
state of the environment, namely, using Q-learning. Although a policy is
not required since it's obtained from the Q-value function, Q-learning is
limited to act in discrete action spaces. For continuous spaces see DDPQ.
DDQN has two important ingredients:
    - A target network. Although the online network makes the predictions,
    the target network helps in the training of the online network. The
    parameters of the target networks are copied every tau steps from the
    online network and kept fixed on all other steps. This reduces the
    overestimations produced in Q-learning.
    - Experience replay. The observed transitions are stored for some time
    and sampled uniformly from a memory bank to update the network.
"""

import lasagne.init as li
import lasagne.layers as ll
import lasagne.nonlinearities as lnl
import lasagne.objectives as lo
import lasagne.updates as lu
import numpy as np
import theano
import theano.tensor as tensor

from garage.algos import RLAlgorithm
from garage.misc.overrides import overrides


class DDQN(RLAlgorithm):
    """Implementation of the DDQN algorithm."""

    def __init__(self,
                 env,
                 exploration_strategy,
                 episodes=1000,
                 tau=64,
                 batch_size=32,
                 exploration_steps=32,
                 gamma=0.91,
                 alpha=0.002,
                 memory_bank_size=10000,
                 units_per_layer=(32, 32),
                 weights_per_layer=(li.GlorotUniform(), li.GlorotUniform()),
                 bias_per_layer=(li.Constant(0.), li.Constant(0.)),
                 nonlinearity_per_layer=(lnl.rectify, lnl.rectify)):
        """Initialize all parameters for DDQN.

        The parameters that define the layers in the neural networks must have
        the same length.

        Parameters
        ----------
        env : environment
            The environment to train the algorithm.
        exploration_strategy : ExplorationStrategy
            Decides the exploitation or exploration of actions.
        episodes : int
            Number of episodes to run the training
        tau : int
            The parameters of the target network are copied every tau steps
            from the online network once the learning of the online neural
            network has started.
        batch_size : int
            Number of observed transitions randomly sampled from the memory
            bank for the training of the online neural network.
        exploration_steps : int
            Number of steps required before starting the learning of the online
            neural network. This must be greater or equal than the batch size
            to save enough observed transitions in the memory bank.
        gamma : float
            Discount factor in the Q-value update. This value has to be in the
            range of [0.0, 1.0]. A value of 0 makes the agent only consider
            immediate rewards, while a value of 1 makes the agent only consider
            later rewards.
        alpha : float
            Learning_rate in the Q-value update. This value has to be in the
            range of [0.0, 1.0]. Exploitation of prior knowledge is used when
            alpha is zero, and exploration when alpha is one.
        memory_bank_size : int
            Maximum number of transitions that will be saved in the memory
            bank. Oldest transitions will be discarded once memory is full.
            If set as -1, memory is not limited.
        units_per_layer : tuple of ints
            Number of units for each layer in the neural network.
        weights_per_layer : tuple of weights
            Initial value of weights for each layer in the neural network.
        bias_per_layer : tuple of biases
            Initial value of biases for each layer in the neural network.
        nonlinearity_per_layer : tuple of non-linearities
            Non-linearities applied to the activations of each layer in the
            neural network.

        """
        assert episodes > 0
        assert tau > 0
        assert batch_size > 0
        assert exploration_steps >= batch_size
        assert (1.0 >= gamma and gamma >= 0.0)
        assert (1.0 >= alpha and alpha >= 0.0)
        assert ((len(units_per_layer) == len(weights_per_layer)) and
                (len(weights_per_layer) == len(bias_per_layer)) and
                (len(bias_per_layer) ==
                 len(nonlinearity_per_layer)))  # yapf: disable
        self._env = env
        self._exploration_strategy = exploration_strategy
        self._episodes = episodes
        self._tau = tau
        self._batch_size = batch_size
        self._exploration_steps = exploration_steps
        self._gamma = gamma
        self._alpha = alpha
        self._units = units_per_layer
        self._weights = weights_per_layer
        self._bias = bias_per_layer
        self._nonlinearities = nonlinearity_per_layer

        self._step_total = 0
        self._render = False
        self._continue_trainig = True
        # Observers of the training
        self._training_obs = []

        self._memory = MemoryBank()
        # Placeholders for variables used in the neural networks
        self._obs = tensor.matrix("Observation")
        self._obs_next = tensor.matrix("ObservationNext")
        self._action = tensor.ivector("Action")
        self._reward = tensor.vector("Reward")
        self._done = tensor.vector("Done")
        # Build the networks
        self._target_nn = self._create_network()
        self._online_nn = self._create_network()
        self._build_graph()

    def _create_network(self):
        """Create a neural network.

        The network consists of one input layer and a variable amount of dense
        layers as specified by parameters passed to the constructor of this
        class.
        The first dimension in the input layer indicates the batch size, but
        it's left as variable (None) since the network is used with the batch
        size for training, and with just one input when doing predictions.
        """
        network = ll.InputLayer((None, ) + self._env.observation_space.shape)
        for u, w, b, n in zip(self._units, self._weights, self._bias,
                              self._nonlinearities):
            network = ll.DenseLayer(network, u, w, b, n)
        network = ll.DenseLayer(network, self._env.action_space.n)
        return network

    def _build_graph(self):
        """Build graph of neural networks.

        The first half of the code builds the graph to calculate the old and
        target values of the Q function based on the target equation for DDQN.
        The second part builds the graph to perform the training of the online
        network based on the error between the old and target value, as well
        as the function to do predictions.
        """
        # Get the old quality value
        q = ll.get_output(
            self._online_nn, deterministic=True, inputs=self._obs)
        old_q_value = q[tensor.arange(q.shape[0]), self._action]
        # Get the best action for the next observation
        q_next = ll.get_output(
            self._online_nn, deterministic=True, inputs=self._obs_next)
        best_action = tensor.argmax(q_next, axis=1)
        # Get the target quality value based on the best action
        q_target_next = ll.get_output(
            self._target_nn, deterministic=True, inputs=self._obs_next)
        best_q_value = q_target_next[tensor.arange(q_target_next.shape[0]),
                                     best_action]
        target_value = (self._reward + self._gamma * self._done * best_q_value)

        # Build loss expression
        loss = lo.squared_error(old_q_value, target_value)
        loss = lo.aggregate(loss, mode="mean")
        # Build the update expression for the online network parameters
        params = ll.get_all_params(self._online_nn, trainable=True)
        updates = lu.sgd(loss, params, self._alpha)
        # Build the functions to train the network and evaluate observations
        self._learn = theano.function(
            [
                self._obs, self._obs_next, self._action, self._reward,
                self._done
            ],
            loss,
            updates=updates)
        self._evaluate = theano.function([self._obs], q)

    def get_action(self, obs):
        """Get the best action based on the prediction of the Q function.

        Parameters
        ----------
        obs : observation
            Current observation of the environment.

        """
        qvals = self._evaluate(obs)
        a = np.argmax(qvals)
        return a

    @overrides
    def train(self):
        """Training for all episodes."""
        try:
            for n in range(self._episodes):
                self._train_per_episode(n)
        finally:
            self._env.close()

    def _train_per_episode(self, n):
        """Training for a single episode."""
        observation = self._env.reset()
        done = False
        reward_episode = 0
        while not done:
            action = self._exploration_strategy.get_action(observation, self)
            self._step_total += 1
            next_observation, reward, done, _ = self._env.step(action)
            self._memory.store(observation, action, next_observation, reward,
                               not done)
            if self._step_total >= self._exploration_steps:
                batch = self._memory.get_batch(self._batch_size)
                self._learn(batch.obs, batch.obs_next, batch.actions,
                            batch.rewards, batch.not_done)
                if not (self._step_total % self._tau):
                    self._copy_to_target()
            observation = next_observation
            reward_episode += reward

    def _copy_to_target(self):
        """Copy weights from online network to target network."""
        ll.set_all_param_values(self._target_nn,
                                ll.get_all_param_values(self._online_nn))


class MemoryBank:
    """The memory bank is used for experience replay in DDQN."""

    def __init__(self, memory_size=10000):
        """Initialize lists and transition counter.

        Parameters
        ----------
        memory_size : int
            Amount of transitions that will be saved in memory. If it's value
            is -1, the memory is not limited, otherwise half of the transitions
            will be removed once memory is full.

        """
        assert (memory_size == -1 or memory_size > 0)
        self._obs_arr = []
        self._obs_next_arr = []
        self._action_arr = []
        self._reward_arr = []
        self._not_done_arr = []
        self._num_transitions = 0
        self._memory_size = memory_size

    def store(self, obs, action, obs_next, reward, not_done):
        """Save a transition in the memory bank."""
        self._obs_arr.append(obs)
        self._obs_next_arr.append(obs_next)
        self._action_arr.append(action)
        self._reward_arr.append(reward)
        self._not_done_arr.append(not_done)
        self._num_transitions += 1
        # Discard oldest half of the memory once it's full
        if (self._memory_size > 0 and
            self._num_transitions == self._memory_size):  # yapf: disable
            half_trans = int(self._num_transitions / 2)
            self._obs_arr = self._obs_arr[half_trans:self._num_transitions]
            self._obs_next_arr = self._obs_next_arr[half_trans:
                                                    self._num_transitions]
            self._action_arr = self._action_arr[half_trans:
                                                self._num_transitions]
            self._reward_arr = self._reward_arr[half_trans:
                                                self._num_transitions]
            self._not_done_arr = self._not_done_arr[half_trans:
                                                    self._num_transitions]
            self._num_transitions -= half_trans

    def get_batch(self, n):
        """Get a batch from the memory bank.

        Parameters
        ----------
        n : int
            Number of transitions to put in the memory batch.

        """
        random_indices = np.random.randint(0, self._num_transitions, n)
        obs = [self._obs_arr[index] for index in random_indices]
        obs_next = [self._obs_next_arr[index] for index in random_indices]
        actions = [self._action_arr[index] for index in random_indices]
        rewards = [self._reward_arr[index] for index in random_indices]
        not_done = [self._not_done_arr[index] for index in random_indices]
        return MemoryBatch(obs, obs_next, actions, rewards, not_done)


class MemoryBatch:
    """Container class for a batch obtained from the memory bank."""

    def __init__(self, obs, obs_next, actions, rewards, not_done):
        """Initialize all sub-batches."""
        self._obs = obs
        self._obs_next = obs_next
        self._actions = actions
        self._rewards = rewards
        self._not_done = not_done

    @property
    def obs(self):
        """Batch of observations."""
        return self._obs

    @property
    def obs_next(self):
        """Batch of next observations."""
        return self._obs_next

    @property
    def actions(self):
        """Batch of actions."""
        return self._actions

    @property
    def rewards(self):
        """Batch of rewards."""
        return self._rewards

    @property
    def not_done(self):
        """Batch of non-termination of episode."""
        return self._not_done
