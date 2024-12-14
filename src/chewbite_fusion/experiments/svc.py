import logging

from sklearn.svm import SVC

from chewbite_fusion.experiments.settings import random_seed
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features.feature_factories import FeatureFactory_v5

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_v1_model_instance(variable_params):
    return SVC(random_state=random_seed,
               **variable_params)


MODEL_PARAMS_GRID = {
    'kernel': ['poly', 'rbf'],
    'class_weight': ['balanced', None]
}


@experiment()
def svc_v1():
    """ Initial SVC experiment FF v5 with window width 0.3 and overlap 0.0.
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
                   name='svc_v1',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def svc_v2():
    """ Initial SVC experiment FF v5 with window width 0.3 and overlap 0.5.
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
                   name='svc_v2',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def svc_v3():
    """ Initial SVC experiment FF v5 with window width 0.5 and overlap 0.0.
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
                   name='svc_v3',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def svc_v4():
    """ Initial SVC experiment FF v5 with window width 0.5 and overlap 0.5.
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
                   name='svc_v4',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def svc_v5():
    """ Initial SVC experiment FF v5 with window width 1.0 and overlap 0.0.
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
                   name='svc_v5',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def svc_v6():
    """ Initial SVC experiment FF v5 with window width 1.0 and overlap 0.5.
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
                   name='svc_v6',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()
