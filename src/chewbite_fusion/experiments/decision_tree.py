import logging

from sklearn.tree import DecisionTreeClassifier

from chewbite_fusion.experiments.settings import random_seed
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features.feature_factories import FeatureFactory_v5

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_v1_model_instance(variable_params):
    return DecisionTreeClassifier(random_state=random_seed,
                                  **variable_params)


MODEL_PARAMS_GRID = {
    'max_depth': [5, 10, 20, None],
    'class_weight': ['balanced', None]
}


@experiment()
def dt_v1():
    """ Initial DT experiment FF v5 with window width 0.3 and overlap 0.0.
    """
    window_width = 0.3
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True)

    e = Experiment(get_v1_model_instance,
                   FeatureFactory_v5,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='dt_v1',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def dt_v2():
    """ Initial DT experiment FF v5 with window width 0.3 and overlap 0.5.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True)

    e = Experiment(get_v1_model_instance,
                   FeatureFactory_v5,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='dt_v2',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def dt_v3():
    """ Initial DT experiment FF v5 with window width 0.5 and overlap 0.0.
    """
    window_width = 0.5
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True)

    e = Experiment(get_v1_model_instance,
                   FeatureFactory_v5,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='dt_v3',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def dt_v4():
    """ Initial DT experiment FF v5 with window width 0.5 and overlap 0.5.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True)

    e = Experiment(get_v1_model_instance,
                   FeatureFactory_v5,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='dt_v4',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def dt_v5():
    """ Initial DT experiment FF v5 with window width 1.0 and overlap 0.0.
    """
    window_width = 1.0
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True)

    e = Experiment(get_v1_model_instance,
                   FeatureFactory_v5,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='dt_v5',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def dt_v6():
    """ Initial DT experiment FF v5 with window width 1.0 and overlap 0.5.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True)

    e = Experiment(get_v1_model_instance,
                   FeatureFactory_v5,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='dt_v6',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()
