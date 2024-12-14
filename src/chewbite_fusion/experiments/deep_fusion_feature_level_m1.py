import logging

from chewbite_fusion.models import deep_fusion_feature_level as dfflm
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features import feature_factories as ff

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_model_instance(variable_params):
    return dfflm.DeepFusionFeatureLevel_m1(
        input_size_imu=variable_params['input_size_imu'],
        input_size_audio=variable_params['input_size_audio'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True)


@experiment()
def deep_fusion_feature_level_m1_v1():
    """ Experiment with fusion network at feature level with window width 0.3s,
        using audio, accelerometer and gyroscope magnitude vectors.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrVectorsRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_feature_level_m1_v1',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_imu': [(None, 30, 2)],
                                          'input_size_audio': [(None, 1800, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m1_v2():
    """ Experiment with fusion network at feature level with window width 0.5s,
        using audio, accelerometer and gyroscope magnitude vectors.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrVectorsRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_feature_level_m1_v2',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_imu': [(None, 50, 2)],
                                          'input_size_audio': [(None, 3000, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m1_v3():
    """ Experiment with fusion network at feature level with window width 1s,
        using audio, accelerometer and gyroscope magnitude vectors.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrVectorsRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_feature_level_m1_v3',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_imu': [(None, 100, 2)],
                                          'input_size_audio': [(None, 6000, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m1_v4():
    """ Experiment with fusion network at feature level with window width 0.3s,
        using raw data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AllRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_feature_level_m1_v4',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_imu': [(None, 30, 9)],
                                          'input_size_audio': [(None, 1800, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m1_v5():
    """ Experiment with fusion network at feature level with window width 0.5s,
        using raw data.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AllRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_feature_level_m1_v5',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_imu': [(None, 50, 9)],
                                          'input_size_audio': [(None, 3000, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m1_v6():
    """ Experiment with fusion network at feature level with window width 1s,
        using raw data.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AllRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_feature_level_m1_v6',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_imu': [(None, 100, 9)],
                                          'input_size_audio': [(None, 6000, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m1_v7():
    """ Experiment with fusion network at feature level with window width 0.3s,
        using audio, accelerometer, gyroscope raw data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_feature_level_m1_v7',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_imu': [(None, 30, 6)],
                                          'input_size_audio': [(None, 1800, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m1_v8():
    """ Experiment with fusion network at feature level with window width 0.5s,
        using audio, accelerometer, gyroscope raw data.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_feature_level_m1_v8',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_imu': [(None, 50, 6)],
                                          'input_size_audio': [(None, 3000, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m1_v9():
    """ Experiment with fusion network at feature level with window width 1s,
        using audio, accelerometer, gyroscope raw data.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AudioAccGyrRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='deep_fusion_feature_level_m1_v9',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_imu': [(None, 100, 6)],
                                          'input_size_audio': [(None, 6000, 1)]})

    e.run()
