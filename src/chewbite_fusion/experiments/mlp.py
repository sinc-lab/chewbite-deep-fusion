import logging

from sklearn.neural_network import MLPClassifier

from chewbite_fusion.experiments.settings import random_seed
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features import feature_factories

from yaer.base import experiment


logger = logging.getLogger('yaer')


MODEL_PARAMS_GRID = {
    'hidden_layer_sizes': [(50), (100),
                           (50, 10), (50, 50), (100, 100),
                           (100, 100, 10), (200, 200, 50)],
    'learning_rate': ['adaptive', 'constant']
}


def get_v1_model_instance(variable_params):
    return MLPClassifier(random_state=random_seed,
                         **variable_params)


@experiment()
def mlp_v1():
    """ Initial MLP experiment with FF v5 with window width 0.3 and overlap 0.0.
    """
    window_width = 0.3
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True)

    e = Experiment(get_v1_model_instance,
                   feature_factories.FeatureFactory_v5,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='mlp_v1',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def mlp_v2():
    """ Initial MLP experiment with FF v5 with window width 0.3 and overlap 0.5.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True)

    e = Experiment(get_v1_model_instance,
                   feature_factories.FeatureFactory_v5,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='mlp_v2',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def mlp_v3():
    """ Initial MLP experiment with FF v5 with window width 0.5 and overlap 0.0.
    """
    window_width = 0.5
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True)

    e = Experiment(get_v1_model_instance,
                   feature_factories.FeatureFactory_v5,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='mlp_v3',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def mlp_v4():
    """ Initial MLP experiment with FF v5 with window width 0.5 and overlap 0.5.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True)

    e = Experiment(get_v1_model_instance,
                   feature_factories.FeatureFactory_v5,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='mlp_v4',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def mlp_v5():
    """ Initial MLP experiment with FF v5 with window width 1.0 and overlap 0.0.
    """
    window_width = 1.0
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True)

    e = Experiment(get_v1_model_instance,
                   feature_factories.FeatureFactory_v5,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='mlp_v5',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def mlp_v6():
    """ Initial MLP experiment with FF v5 with window width 1.0 and overlap 0.5.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True)

    e = Experiment(get_v1_model_instance,
                   feature_factories.FeatureFactory_v5,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='mlp_v6',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()
