import logging

from xgboost import XGBClassifier

from chewbite_fusion.experiments.settings import random_seed
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features import feature_factories

from yaer.base import experiment


logger = logging.getLogger('yaer')


MODEL_PARAMS_GRID = {
    'max_depth': [5, 10, 20, None],
    'n_estimators': [10, 50, 100],
    'max_leaves': [20, 50, 100, None]
}


def get_v1_model_instance(variable_params):
    return XGBClassifier(random_state=random_seed,
                         n_jobs=-1,
                         **variable_params)


@experiment()
def xgb_v1():
    """ Initial XGBoost experiment with FF v5, window width 0.3 and overlap 0.0.
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
                   name='xgb_v1',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def xgb_v2():
    """ Initial XGBoost experiment with FF v5, window width 0.3 and overlap 0.5.
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
                   name='xgb_v2',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def xgb_v3():
    """ Initial XGBoost experiment with FF v5, window width 0.5 and overlap 0.0.
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
                   name='xgb_v3',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def xgb_v4():
    """ Initial XGBoost experiment with FF v5, window width 0.5 and overlap 0.5.
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
                   name='xgb_v4',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def xgb_v5():
    """ Initial XGBoost experiment with FF v5, window width 1.0 and overlap 0.0.
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
                   name='xgb_v5',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()


@experiment()
def xgb_v6():
    """ Initial XGBoost experiment with FF v5, window width 1.0 and overlap 0.5.
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
                   name='xgb_v6',
                   manage_sequences=False,
                   model_parameters_grid=MODEL_PARAMS_GRID)

    e.run()
