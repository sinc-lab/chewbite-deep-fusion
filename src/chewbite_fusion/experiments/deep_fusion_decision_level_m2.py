import logging
import functools
from sklearn.feature_selection import SelectFromModel

from yaer.base import experiment
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from chewbite_fusion.models import deep_fusion_decision_level as dfdl
from chewbite_fusion.models.deep_sound import DeepSound
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features import feature_factories as ff
from chewbite_fusion.experiments.settings import random_seed


logger = logging.getLogger('yaer')


def get_alvarenga_model_instance():
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


def get_model_instance_dt(variable_params):
    deep_sound_model = functools.partial(DeepSound,
                                         10,                                    # batch_size
                                         variable_params['input_size_audio'],   # input_size
                                         5,                                     # output_size
                                         1500,                                  # n_epochs
                                         True,                                  # training_reshape
                                         True,                                  # set_sample_weights
                                         True)                                  # feature_scaling

    def final_decision_model():
        return DecisionTreeClassifier(max_depth=3,
                                      random_state=random_seed,
                                      class_weight='balanced')

    t = dfdl.DeepFusionDecisionLevel_m2(sound_model_factory=deep_sound_model,
                                        movement_model_factory=get_alvarenga_model_instance,
                                        decision_model_factory=final_decision_model)

    return t


def get_model_instance_mlp(variable_params):
    deep_sound_model = functools.partial(DeepSound,
                                         10,                                    # batch_size
                                         variable_params['input_size_audio'],   # input_size
                                         5,                                     # output_size
                                         1500,                                  # n_epochs
                                         True,                                  # training_reshape
                                         True,                                  # set_sample_weights
                                         True)                                  # feature_scaling

    def final_decision_model():
        return MLPClassifier(hidden_layer_sizes=(50),
                             random_state=random_seed)

    t = dfdl.DeepFusionDecisionLevel_m2(sound_model_factory=deep_sound_model,
                                        movement_model_factory=get_alvarenga_model_instance,
                                        decision_model_factory=final_decision_model)

    return t


@experiment()
def deep_fusion_decision_level_m2_v1():
    """ Experiment with fusion network at decision level with window width 0.3s.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance_dt,
                   ff.FeatureFactory_DecisionLevelMixData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_decision_level_m2_v1',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_audio': [1800]})

    e.run()


@experiment()
def deep_fusion_decision_level_m2_v2():
    """ Experiment with fusion network at decision level with window width 0.5s.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance_dt,
                   ff.FeatureFactory_DecisionLevelMixData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_decision_level_m2_v2',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_audio': [3000]})

    e.run()


@experiment()
def deep_fusion_decision_level_m2_v3():
    """ Experiment with fusion network at decision level with window width 1s.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance_dt,
                   ff.FeatureFactory_DecisionLevelMixData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_decision_level_m2_v3',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_audio': [6000]})

    e.run()


@experiment()
def deep_fusion_decision_level_m2_v4():
    """ Experiment with fusion network at decision level with window width 0.3s.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance_mlp,
                   ff.FeatureFactory_DecisionLevelMixData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_decision_level_m2_v4',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_audio': [1800]})

    e.run()


@experiment()
def deep_fusion_decision_level_m2_v5():
    """ Experiment with fusion network at decision level with window width 0.5s.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance_mlp,
                   ff.FeatureFactory_DecisionLevelMixData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_decision_level_m2_v5',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_audio': [3000]})

    e.run()


@experiment()
def deep_fusion_decision_level_m2_v6():
    """ Experiment with fusion network at decision level with window width 1s.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance_mlp,
                   ff.FeatureFactory_DecisionLevelMixData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_decision_level_m2_v6',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_audio': [6000]})

    e.run()
