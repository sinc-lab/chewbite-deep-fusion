import logging

from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.models.random_model import RandomEstimator
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features.feature_factories import FeatureFactory_v1

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_model_instance(variable_params):
    return RandomEstimator()


@experiment()
def random_experiment():
    """ Random baseline experiment.
    """
    X, y = main()

    e = Experiment(get_model_instance,
                   FeatureFactory_v1,
                   X, y,
                   window_width=0.5,
                   window_overlap=0.5,
                   name='random_experiment',
                   manage_sequences=False)

    e.run()


@experiment()
def random_experiment_v1():
    """ Random baseline experiment v1.
    """
    X, y = main(movement_sampling_frequency=50,
                window_width=0.5,
                window_overlap=0.0)

    e = Experiment(get_model_instance,
                   FeatureFactory_v1,
                   X, y,
                   window_width=0.5,
                   window_overlap=0.0,
                   name='random_experiment_v1',
                   movement_sampling_frequency=50,
                   manage_sequences=False)

    e.run()
