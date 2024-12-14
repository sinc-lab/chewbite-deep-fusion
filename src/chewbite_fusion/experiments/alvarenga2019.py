import logging

from scipy import signal
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline


from chewbite_fusion.experiments.settings import random_seed
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features.feature_factories import FeatureFactory_Alvarenga2019

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_model_instance(variable_params):
    feature_importance_estimator = RandomForestClassifier(
        n_estimators=500,
        max_features='sqrt',
        random_state=random_seed,
        n_jobs=-1
    )
    feature_selector = SelectFromModel(feature_importance_estimator,
                                       max_features=5)

    pipe = Pipeline([
        ('feature_selector', feature_selector),
        ('classifier', DecisionTreeClassifier(random_state=random_seed))
    ])

    return pipe


@experiment()
def alvarenga2019():
    """ Implementation based on Alvarenga 2019.
    """
    window_width = 1.0
    window_overlap = 0.5

    X, y = main(include_movement_magnitudes=True,
                window_width=window_width,
                window_overlap=window_overlap)

    e = Experiment(get_model_instance,
                   FeatureFactory_Alvarenga2019,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='alvarenga2019',
                   manage_sequences=False)

    e.run()


@experiment()
def alvarenga2019_v1():
    """ Implementation based on Alvarenga 2019.
    """
    window_width = 0.3
    window_overlap = 0.0

    X, y = main(include_movement_magnitudes=True,
                window_width=window_width,
                window_overlap=window_overlap)

    e = Experiment(get_model_instance,
                   FeatureFactory_Alvarenga2019,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='alvarenga2019_v1',
                   manage_sequences=False)

    e.run()


@experiment()
def alvarenga2019_v2():
    """ Implementation based on Alvarenga 2019.
    """
    window_width = 0.3
    window_overlap = 0.5

    X, y = main(include_movement_magnitudes=True,
                window_width=window_width,
                window_overlap=window_overlap)

    e = Experiment(get_model_instance,
                   FeatureFactory_Alvarenga2019,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='alvarenga2019_v2',
                   manage_sequences=False)

    e.run()


@experiment()
def alvarenga2019_v3():
    """ Implementation based on Alvarenga 2019.
    """
    window_width = 0.5
    window_overlap = 0.0

    X, y = main(include_movement_magnitudes=True,
                window_width=window_width,
                window_overlap=window_overlap)

    e = Experiment(get_model_instance,
                   FeatureFactory_Alvarenga2019,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='alvarenga2019_v3',
                   manage_sequences=False)

    e.run()


@experiment()
def alvarenga2019_v4():
    """ Implementation based on Alvarenga 2019.
    """
    window_width = 0.5
    window_overlap = 0.5

    X, y = main(include_movement_magnitudes=True,
                window_width=window_width,
                window_overlap=window_overlap)

    e = Experiment(get_model_instance,
                   FeatureFactory_Alvarenga2019,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='alvarenga2019_v4',
                   manage_sequences=False)

    e.run()


@experiment()
def alvarenga2019_v5():
    """ Implementation based on Alvarenga 2019.
    """
    window_width = 1.0
    window_overlap = 0.0

    X, y = main(include_movement_magnitudes=True,
                window_width=window_width,
                window_overlap=window_overlap)

    e = Experiment(get_model_instance,
                   FeatureFactory_Alvarenga2019,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='alvarenga2019_v5',
                   manage_sequences=False)

    e.run()


@experiment()
def alvarenga2019_v6():
    """ Implementation based on Alvarenga 2019.
    """
    window_width = 1.0
    window_overlap = 0.5

    X, y = main(include_movement_magnitudes=True,
                window_width=window_width,
                window_overlap=window_overlap)

    e = Experiment(get_model_instance,
                   FeatureFactory_Alvarenga2019,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='alvarenga2019_v6',
                   manage_sequences=False)

    e.run()
