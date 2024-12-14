import logging

from chewbite_fusion.models import deep_fusion_feature_level as dfflm
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features import feature_factories as ff

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_model_instance(variable_params):
    return dfflm.DeepFusionFeatureLevel_m7(
        input_size_acc=variable_params['input_size_acc'],
        input_size_gyr=variable_params['input_size_gyr'],
        input_size_mag=variable_params['input_size_mag'],
        input_size_audio=variable_params['input_size_audio'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True)


@experiment()
def deep_fusion_feature_level_m7_v1():
    """ Experiment with fusion network (3 heads CNN) at feature level with window width 0.3s,
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
                   name='deep_fusion_feature_level_m7_v1',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 30, 1)],
                                          'input_size_gyr': [(None, 30, 1)],
                                          'input_size_mag': [None],
                                          'input_size_audio': [(None, 1800, 1)]})

    e.run()

@experiment()
def deep_fusion_feature_level_m7_v2():
    """ Experiment with fusion network (3 heads CNN) at feature level with window width 0.5s,
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
                   name='deep_fusion_feature_level_m7_v2',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 50, 1)],
                                          'input_size_gyr': [(None, 50, 1)],
                                          'input_size_mag': [None],
                                          'input_size_audio': [(None, 3000, 1)]})

    e.run()

@experiment()
def deep_fusion_feature_level_m7_v3():
    """ Experiment with fusion network (3 heads CNN) at feature level with window width 1.0s,
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
                   name='deep_fusion_feature_level_m7_v3',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 100, 1)],
                                          'input_size_gyr': [(None, 100, 1)],
                                          'input_size_mag': [None],
                                          'input_size_audio': [(None, 6000, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m7_v4():
    """ Experiment with fusion network (3 heads CNN) at feature level with window width 0.3s,
        using audio, accelerometer, gyroscope and magnetomer raw data.
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
                   name='deep_fusion_feature_level_m7_v4',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 30, 3)],
                                          'input_size_gyr': [(None, 30, 3)],
                                          'input_size_mag': [(None, 30, 3)],
                                          'input_size_audio': [(None, 1800, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m7_v5():
    """ Experiment with fusion network (3 heads CNN) at feature level with window width 0.5s,
        using audio, accelerometer, gyroscope and magnetometer raw data.
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
                   name='deep_fusion_feature_level_m7_v5',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 50, 3)],
                                          'input_size_gyr': [(None, 50, 3)],
                                          'input_size_mag': [(None, 50, 3)],
                                          'input_size_audio': [(None, 3000, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m7_v6():
    """ Experiment with fusion network (3 heads CNN) at feature level with window width 1.0s,
        using audio, accelerometer, gyroscope and magnetometer raw data.
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
                   name='deep_fusion_feature_level_m7_v6',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 100, 3)],
                                          'input_size_gyr': [(None, 100, 3)],
                                          'input_size_mag': [(None, 100, 3)],
                                          'input_size_audio': [(None, 6000, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m7_v7():
    """ Experiment with fusion network (3 heads CNN) at feature level with window width 0.3s,
        using audio, accelerometer and gyroscope raw data.
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
                   name='deep_fusion_feature_level_m7_v7',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 30, 3)],
                                          'input_size_gyr': [(None, 30, 3)],
                                          'input_size_mag': [None],
                                          'input_size_audio': [(None, 1800, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m7_v8():
    """ Experiment with fusion network (3 heads CNN) at feature level with window width 0.5s,
        using audio, accelerometer and gyroscope raw data.
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
                   name='deep_fusion_feature_level_m7_v8',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 50, 3)],
                                          'input_size_gyr': [(None, 50, 3)],
                                          'input_size_mag': [None],
                                          'input_size_audio': [(None, 3000, 1)]})

    e.run()


@experiment()
def deep_fusion_feature_level_m7_v9():
    """ Experiment with fusion network (3 heads CNN) at feature level with window width 1.0s,
        using audio, accelerometer and gyroscope raw data.
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
                   name='deep_fusion_feature_level_m7_v9',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 100, 3)],
                                          'input_size_gyr': [(None, 100, 3)],
                                          'input_size_mag': [None],
                                          'input_size_audio': [(None, 6000, 1)]})

    e.run()