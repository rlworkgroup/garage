import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.q_functions import ContinuousCNNQFunction
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDictEnv
from tests.fixtures.envs.dummy import DummyDiscreteEnv
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv
from tests.fixtures.models import SimpleCNNModel
from tests.fixtures.models import SimpleCNNModelWithMaxPooling
from tests.fixtures.models import SimpleMLPMergeModel


class TestContinuousCNNQFunction(TfGraphTestCase):

    # yapf: disable
    @pytest.mark.parametrize('filter_dims, num_filters, strides', [
        ((3, ), (5, ), (1, )),
        ((3, ), (5, ), (2, )),
        ((3, 3), (5, 5), (1, 1)),
    ])
    # yapf: enable
    def test_get_qval(self, filter_dims, num_filters, strides):
        env = TfEnv(DummyDiscretePixelEnv())
        obs = env.reset()

        with mock.patch(('garage.tf.models.'
                         'cnn_mlp_merge_model.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.models.'
                             'cnn_mlp_merge_model.MLPMergeModel'),
                            new=SimpleMLPMergeModel):
                qf = ContinuousCNNQFunction(env_spec=env.spec,
                                            filter_dims=filter_dims,
                                            num_filters=num_filters,
                                            strides=strides)

        action_dim = env.action_space.shape

        obs, _, _, _ = env.step(1)

        act = np.full(action_dim, 0.5)
        expected_output = np.full((1, ), 0.5)

        outputs = qf.get_qval([obs], [act])

        assert np.array_equal(outputs[0], expected_output)

        outputs = qf.get_qval([obs, obs, obs], [act, act, act])

        for output in outputs:
            assert np.array_equal(output, expected_output)

        # make sure observations are unflattened

        obs = env.observation_space.flatten(obs)
        qf._f_qval = mock.MagicMock()

        qf.get_qval([obs], [act])
        unflattened_obs = qf._f_qval.call_args_list[0][0][0]
        assert unflattened_obs.shape[1:] == env.spec.observation_space.shape

        qf.get_qval([obs, obs], [act, act])
        unflattened_obs = qf._f_qval.call_args_list[1][0][0]
        assert unflattened_obs.shape[1:] == env.spec.observation_space.shape

    # yapf: disable
    @pytest.mark.parametrize('filter_dims, num_filters, strides, '
                             'pool_strides, pool_shapes', [
                                ((3, ), (5, ), (1, ), (1, 1), (1, 1)),
                                ((3, ), (5, ), (2, ), (2, 2), (2, 2)),
                                ((3, 3), (5, 5), (1, 1), (1, 1), (1, 1)),
                                ((3, 3), (5, 5), (1, 1), (2, 2), (2, 2))])
    # yapf: enable
    def test_get_qval_max_pooling(self, filter_dims, num_filters, strides,
                                  pool_strides, pool_shapes):
        env = TfEnv(DummyDiscretePixelEnv())
        obs = env.reset()

        with mock.patch(('garage.tf.models.'
                         'cnn_mlp_merge_model.CNNModelWithMaxPooling'),
                        new=SimpleCNNModelWithMaxPooling):
            with mock.patch(('garage.tf.models.'
                             'cnn_mlp_merge_model.MLPMergeModel'),
                            new=SimpleMLPMergeModel):
                qf = ContinuousCNNQFunction(env_spec=env.spec,
                                            filter_dims=filter_dims,
                                            num_filters=num_filters,
                                            strides=strides,
                                            max_pooling=True,
                                            pool_strides=pool_strides,
                                            pool_shapes=pool_shapes)

        action_dim = env.action_space.shape

        obs, _, _, _ = env.step(1)

        act = np.full(action_dim, 0.5)
        expected_output = np.full((1, ), 0.5)

        outputs = qf.get_qval([obs], [act])

        assert np.array_equal(outputs[0], expected_output)

        outputs = qf.get_qval([obs, obs, obs], [act, act, act])

        for output in outputs:
            assert np.array_equal(output, expected_output)

    # yapf: disable
    @pytest.mark.parametrize('obs_dim', [
                                    (1, ),
                                    (1, 1, 1, 1),
                                    (2, 2, 2, 2)])
    # yapf: enable
    def test_invalid_obs_dim(self, obs_dim):
        with pytest.raises(ValueError):
            env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim))
            ContinuousCNNQFunction(env_spec=env.spec,
                                   filter_dims=(3, ),
                                   num_filters=(5, ),
                                   strides=(1, ))

    def test_not_box(self):
        with pytest.raises(ValueError):
            dict_env = TfEnv(DummyDictEnv())
            ContinuousCNNQFunction(env_spec=dict_env.spec,
                                   filter_dims=(3, ),
                                   num_filters=(5, ),
                                   strides=(1, ))

    def test_obs_is_image(self):
        image_env = TfEnv(DummyDiscretePixelEnv(), is_image=True)
        with mock.patch(('tests.fixtures.models.SimpleCNNModel._build'),
                        autospec=True,
                        side_effect=SimpleCNNModel._build) as build:
            with mock.patch(('garage.tf.models.'
                             'cnn_mlp_merge_model.CNNModel'),
                            new=SimpleCNNModel):
                with mock.patch(('garage.tf.models.'
                                 'cnn_mlp_merge_model.MLPMergeModel'),
                                new=SimpleMLPMergeModel):

                    qf = ContinuousCNNQFunction(env_spec=image_env.spec,
                                                filter_dims=(3, ),
                                                num_filters=(5, ),
                                                strides=(1, ))

                    fake_obs = [
                        np.full(image_env.spec.observation_space.shape, 255)
                    ]

                    # make sure image obses are normalized in _initialize()
                    # and get_qval
                    normalized_obs = build.call_args_list[0][0][1]
                    assert normalized_obs != qf.inputs[0]

                    assert (self.sess.run(normalized_obs,
                                          feed_dict={qf.inputs[0]:
                                                     fake_obs}) == 1.).all()

                    # make sure image obses are normalized in get_qval_sim()

                    obs_dim = image_env.spec.observation_space.shape
                    state_input = tf.compat.v1.placeholder(tf.float32,
                                                           shape=(None, ) +
                                                           obs_dim)

                    act_dim = image_env.spec.observation_space.shape
                    action_input = tf.compat.v1.placeholder(tf.float32,
                                                            shape=(None, ) +
                                                            act_dim)

                    qf.get_qval_sym(state_input, action_input, name='another')
                    normalized_obs = build.call_args_list[1][0][1]

                    assert (self.sess.run(normalized_obs,
                                          feed_dict={state_input:
                                                     fake_obs}) == 1.).all()

    def test_obs_not_image(self):
        env = TfEnv(DummyDiscretePixelEnv())

        with mock.patch(('tests.fixtures.models.SimpleCNNModel._build'),
                        autospec=True,
                        side_effect=SimpleCNNModel._build) as build:
            with mock.patch(('garage.tf.models.'
                             'cnn_mlp_merge_model.CNNModel'),
                            new=SimpleCNNModel):
                with mock.patch(('garage.tf.models.'
                                 'cnn_mlp_merge_model.MLPMergeModel'),
                                new=SimpleMLPMergeModel):

                    qf = ContinuousCNNQFunction(env_spec=env.spec,
                                                filter_dims=(3, ),
                                                num_filters=(5, ),
                                                strides=(1, ))

                    # ensure non-image obses are not normalized
                    # in _initialize() and get_qval()

                    normalized_obs = build.call_args_list[0][0][1]
                    assert normalized_obs == qf.inputs[0]

                    fake_obs = [
                        np.full(env.spec.observation_space.shape, 255.)
                    ]

                    assert (self.sess.run(normalized_obs,
                                          feed_dict={qf.inputs[0]:
                                                     fake_obs}) == 255.).all()

                    # ensure non-image obses are not normalized
                    # in get_qval_sym()

                    obs_dim = env.spec.observation_space.shape
                    state_input = tf.compat.v1.placeholder(tf.float32,
                                                           shape=(None, ) +
                                                           obs_dim)

                    act_dim = env.spec.observation_space.shape
                    action_input = tf.compat.v1.placeholder(tf.float32,
                                                            shape=(None, ) +
                                                            act_dim)

                    qf.get_qval_sym(state_input, action_input, name='another')
                    normalized_obs = build.call_args_list[1][0][1]

                    assert (self.sess.run(normalized_obs,
                                          feed_dict={state_input:
                                                     fake_obs}) == 255.).all()

    # yapf: disable
    @pytest.mark.parametrize('filter_dims, num_filters, strides',
                             [((3, ), (5, ), (1, )),
                              ((3, ), (5, ), (2, )),
                              ((3, 3), (5, 5), (1, 1))])
    # yapf: enable
    def test_get_qval_sym(self, filter_dims, num_filters, strides):
        env = TfEnv(DummyDiscretePixelEnv())
        obs = env.reset()

        with mock.patch(('garage.tf.models.'
                         'cnn_mlp_merge_model.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.models.'
                             'cnn_mlp_merge_model.MLPMergeModel'),
                            new=SimpleMLPMergeModel):
                qf = ContinuousCNNQFunction(env_spec=env.spec,
                                            filter_dims=filter_dims,
                                            num_filters=num_filters,
                                            strides=strides)
        action_dim = env.action_space.shape

        obs, _, _, _ = env.step(1)
        act = np.full(action_dim, 0.5)

        output1 = qf.get_qval([obs], [act])

        input_var1 = tf.compat.v1.placeholder(tf.float32,
                                              shape=(None, ) + obs.shape)
        input_var2 = tf.compat.v1.placeholder(tf.float32,
                                              shape=(None, ) + act.shape)
        q_vals = qf.get_qval_sym(input_var1, input_var2, 'another')

        output2 = self.sess.run(q_vals,
                                feed_dict={
                                    input_var1: [obs],
                                    input_var2: [act]
                                })

        expected_output = np.full((1, ), 0.5)

        assert np.array_equal(output1, output2)
        assert np.array_equal(output2[0], expected_output)

    # yapf: disable
    @pytest.mark.parametrize('filter_dims, num_filters, strides', [
        ((3, ), (5, ), (1, )),
        ((3, ), (5, ), (2, )),
        ((3, 3), (5, 5), (1, 1)),
    ])
    # yapf: enable
    def test_is_pickleable(self, filter_dims, num_filters, strides):

        env = TfEnv(DummyDiscretePixelEnv())
        obs = env.reset()

        with mock.patch(('garage.tf.models.'
                         'cnn_mlp_merge_model.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.models.'
                             'cnn_mlp_merge_model.MLPMergeModel'),
                            new=SimpleMLPMergeModel):
                qf = ContinuousCNNQFunction(env_spec=env.spec,
                                            filter_dims=filter_dims,
                                            num_filters=num_filters,
                                            strides=strides)

        action_dim = env.action_space.shape

        obs, _, _, _ = env.step(1)
        act = np.full(action_dim, 0.5)
        _, _ = qf.inputs

        with tf.compat.v1.variable_scope(
                'ContinuousCNNQFunction/CNNMLPMergeModel/SimpleMLPMergeModel',
                reuse=True):
            return_var = tf.compat.v1.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())

        output1 = qf.get_qval([obs], [act])

        h_data = pickle.dumps(qf)
        with tf.compat.v1.Session(graph=tf.Graph()):
            qf_pickled = pickle.loads(h_data)
            _, _ = qf_pickled.inputs
            output2 = qf_pickled.get_qval([obs], [act])

        assert np.array_equal(output1, output2)

    # yapf: disable
    @pytest.mark.parametrize('filter_dims, num_filters, strides', [
        ((3, ), (5, ), (1, )),
        ((3, ), (5, ), (2, )),
        ((3, 3), (5, 5), (1, 1))
    ])
    # yapf: enable
    def test_clone(self, filter_dims, num_filters, strides):
        env = TfEnv(DummyDiscretePixelEnv())

        with mock.patch(('garage.tf.models.'
                         'cnn_mlp_merge_model.CNNModel'),
                        new=SimpleCNNModel):
            with mock.patch(('garage.tf.models.'
                             'cnn_mlp_merge_model.MLPMergeModel'),
                            new=SimpleMLPMergeModel):
                qf = ContinuousCNNQFunction(env_spec=env.spec,
                                            filter_dims=filter_dims,
                                            num_filters=num_filters,
                                            strides=strides)

                qf_clone = qf.clone('another_qf')

        # pylint: disable=protected-access
        assert qf_clone._filter_dims == qf._filter_dims
        assert qf_clone._num_filters == qf._num_filters
        assert qf_clone._strides == qf._strides
        # pylint: enable=protected-access
