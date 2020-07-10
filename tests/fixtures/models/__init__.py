"""Mock models for testing."""
from tests.fixtures.models.simple_categorical_gru_model import (
    SimpleCategoricalGRUModel)
from tests.fixtures.models.simple_categorical_lstm_model import (
    SimpleCategoricalLSTMModel)
from tests.fixtures.models.simple_categorical_mlp_model import (
    SimpleCategoricalMLPModel)
from tests.fixtures.models.simple_cnn_model import SimpleCNNModel
from tests.fixtures.models.simple_cnn_model_with_max_pooling import (
    SimpleCNNModelWithMaxPooling)
from tests.fixtures.models.simple_gru_model import SimpleGRUModel
from tests.fixtures.models.simple_lstm_model import SimpleLSTMModel
from tests.fixtures.models.simple_mlp_merge_model import SimpleMLPMergeModel
from tests.fixtures.models.simple_mlp_model import SimpleMLPModel

__all__ = [
    'SimpleCategoricalGRUModel',
    'SimpleCategoricalLSTMModel',
    'SimpleCategoricalMLPModel',
    'SimpleCNNModel',
    'SimpleCNNModelWithMaxPooling',
    'SimpleGRUModel',
    'SimpleLSTMModel',
    'SimpleMLPMergeModel',
    'SimpleMLPModel',
]
