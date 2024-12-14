import logging

from chewbite_fusion.models import deep_fusion_feature_level as dffl
from chewbite_fusion.models import deep_fusion_feature_level_with_TL as dffltl
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features import feature_factories as ff
from chewbite_fusion.models.settings import models_path

from yaer.base import experiment


logger = logging.getLogger('yaer')

def get_base_model_instance(variable_params):
    return dffl.DeepFusionFeatureLevel_m7(
        input_size_acc=variable_params['input_size_acc'],
        input_size_gyr=variable_params['input_size_gyr'],
        input_size_audio=variable_params['input_size_audio'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True)


def get_model_instance(variable_params):
    return dffltl.DeepFusionFeatureLevelWithTL_m1(
        input_size_acc=variable_params['input_size_acc'],
        input_size_gyr=variable_params['input_size_gyr'],
        input_size_audio=variable_params['input_size_audio'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True,
        acc_base_model_path=f'{models_path}/acc_cnn_trained_test_v1.keras',
        sound_base_model_path=f'{models_path}/sound_cnn_trained_test_v1.keras',
        layers_to_unfreeze_acc=variable_params['layers_to_unfreeze_acc'],
        layers_to_unfreeze_audio=variable_params['layers_to_unfreeze_audio']
        )


@experiment()
def tl_deep_fusion_feature_level_with_m7_base():
    """ Base experiment with fusion network at feature level using TL for comparison. Model configuration
        with best results is used (DeepFusionFeatureLevel_m7). No transfer learning is used.
    """
    window_width = 0.3
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                movement_sampling_frequency=50)

    e = Experiment(get_base_model_instance,
                   ff.FeatureFactory_AudioAccGyrRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='tl_deep_fusion_feature_level_with_m7_base',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 15, 3)],
                                          'input_size_gyr': [(None, 15, 3)],
                                          'input_size_audio': [(None, 1800, 1)]})

    e.run()


@experiment()
def tl_deep_fusion_feature_level_with_m7_v1():
    """ Experiment with fusion network at feature level using TL. Model configuration with best
        results is used (DeepFusionFeatureLevel_m7). Number of layers to unfreeze is equal to 0.
    """
    window_width = 0.3
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                movement_sampling_frequency=50)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='tl_deep_fusion_feature_level_with_m7_v1',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 15, 3)],
                                          'input_size_gyr': [(None, 15, 3)],
                                          'input_size_audio': [(None, 1800, 1)],
                                          'layers_to_unfreeze_acc': [0],
                                          'layers_to_unfreeze_audio': [0]})

    e.run()


@experiment()
def tl_deep_fusion_feature_level_with_m7_v2():
    """ Experiment with fusion network at feature level using TL. Model configuration with best
        results is used (DeepFusionFeatureLevel_m7). Number of layers to unfreeze is equal to 1.
    """
    window_width = 0.3
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                movement_sampling_frequency=50)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='tl_deep_fusion_feature_level_with_m7_v2',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 15, 3)],
                                          'input_size_gyr': [(None, 15, 3)],
                                          'input_size_audio': [(None, 1800, 1)],
                                          'layers_to_unfreeze_acc': [1],
                                          'layers_to_unfreeze_audio': [1]})

    e.run()


@experiment()
def tl_deep_fusion_feature_level_with_m7_v3():
    """ Experiment with fusion network at feature level using TL. Model configuration with best
        results is used (DeepFusionFeatureLevel_m7). Number of layers to unfreeze is equal to 1 for
        acc and 2 for sound.
    """
    window_width = 0.3
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                movement_sampling_frequency=50)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='tl_deep_fusion_feature_level_with_m7_v3',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 15, 3)],
                                          'input_size_gyr': [(None, 15, 3)],
                                          'input_size_audio': [(None, 1800, 1)],
                                          'layers_to_unfreeze_acc': [1],
                                          'layers_to_unfreeze_audio': [2]})

    e.run()


@experiment()
def tl_deep_fusion_feature_level_with_m7_v4():
    """ Experiment with fusion network at feature level using TL. Model configuration with best
        results is used (DeepFusionFeatureLevel_m7). Number of layers to unfreeze is equal to 1 for
        acc and 3 for sound.
    """
    window_width = 0.3
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                movement_sampling_frequency=50)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='tl_deep_fusion_feature_level_with_m7_v4',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 15, 3)],
                                          'input_size_gyr': [(None, 15, 3)],
                                          'input_size_audio': [(None, 1800, 1)],
                                          'layers_to_unfreeze_acc': [1],
                                          'layers_to_unfreeze_audio': [3]})

    e.run()


@experiment()
def tl_deep_fusion_feature_level_with_m7_v5():
    """ Experiment with fusion network at feature level using TL. Model configuration with best
        results is used (DeepFusionFeatureLevel_m7). All layers are unfreezed.
    """
    window_width = 0.3
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                movement_sampling_frequency=50)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='tl_deep_fusion_feature_level_with_m7_v5',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 15, 3)],
                                          'input_size_gyr': [(None, 15, 3)],
                                          'input_size_audio': [(None, 1800, 1)],
                                          'layers_to_unfreeze_acc': [2],
                                          'layers_to_unfreeze_audio': [6]})

    e.run()


@experiment()
def tl_deep_fusion_feature_level_with_m7_v6():
    """ Experiment with fusion network at feature level using TL. Model configuration with best
        results is used (DeepFusionFeatureLevel_m7). All layers are unfreezed on audio, none of imu.
    """
    window_width = 0.3
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                movement_sampling_frequency=50)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='tl_deep_fusion_feature_level_with_m7_v6',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 15, 3)],
                                          'input_size_gyr': [(None, 15, 3)],
                                          'input_size_audio': [(None, 1800, 1)],
                                          'layers_to_unfreeze_acc': [0],
                                          'layers_to_unfreeze_audio': [6]})

    e.run()


@experiment()
def tl_deep_fusion_feature_level_with_m7_v7():
    """ Experiment with fusion network at feature level using TL. Model configuration with best
        results is used (DeepFusionFeatureLevel_m7). All layers are unfreezed on imu, none of audio.
    """
    window_width = 0.3
    window_overlap = 0.0
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                movement_sampling_frequency=50)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='tl_deep_fusion_feature_level_with_m7_v7',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 15, 3)],
                                          'input_size_gyr': [(None, 15, 3)],
                                          'input_size_audio': [(None, 1800, 1)],
                                          'layers_to_unfreeze_acc': [2],
                                          'layers_to_unfreeze_audio': [0]})

    e.run()
