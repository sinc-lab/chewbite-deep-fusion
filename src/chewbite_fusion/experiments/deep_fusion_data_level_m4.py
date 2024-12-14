import logging

from chewbite_fusion.models import deep_fusion_data_level as dfdlm
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features import feature_factories as ff

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_model_instance(variable_params):
    return dfdlm.DeepFusionDataLevel_m4(
        input_size=variable_params['input_size'],
        output_size=5,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True)


@experiment()
def deep_fusion_data_level_m4_v1():
    """ Experiment with fusion network at data level with window width 0.3s.
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
                   name='deep_fusion_data_level_m4_v1',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(None, 1800, 10)]})

    e.run()


@experiment()
def deep_fusion_data_level_m4_v2():
    """ Experiment with fusion network at data level with window width 0.5s.
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
                   name='deep_fusion_data_level_m4_v2',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(None, 3000, 10)]})

    e.run()


@experiment()
def deep_fusion_data_level_m4_v3():
    """ Experiment with fusion network at data level with window width 1.0s.
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
                   name='deep_fusion_data_level_m4_v3',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(None, 6000, 10)]})

    e.run()


@experiment()
def deep_fusion_data_level_m4_v4():
    """ Experiment with fusion network at data level with window width 0.3s,
        no magnetometer data.
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
                   name='deep_fusion_data_level_m4_v4',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(None, 1800, 7)]})

    e.run()


@experiment()
def deep_fusion_data_level_m4_v5():
    """ Experiment with fusion network at data level with window width 0.5s,
        no magnetometer data.
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
                   name='deep_fusion_data_level_m4_v5',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(None, 3000, 7)]})

    e.run()


@experiment()
def deep_fusion_data_level_m4_v6():
    """ Experiment with fusion network at data level with window width 1.0s,
        no magnetometer data.
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
                   name='deep_fusion_data_level_m4_v6',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(None, 6000, 7)]})

    e.run()


@experiment()
def deep_fusion_data_level_m4_v7():
    """ Experiment with fusion network at data level with window width 0.3s,
        using acc and gyr magnitude vectors.
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
                   name='deep_fusion_data_level_m4_v7',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(None, 1800, 3)]})

    e.run()


@experiment()
def deep_fusion_data_level_m4_v8():
    """ Experiment with fusion network at data level with window width 0.5s,
        using acc and gyr magnitude vectors.
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
                   name='deep_fusion_data_level_m4_v8',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(None, 3000, 3)]})

    e.run()


@experiment()
def deep_fusion_data_level_m4_v9():
    """ Experiment with fusion network at data level with window width 1.0s,
        using acc and gyr magnitude vectors.
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
                   name='deep_fusion_data_level_m4_v9',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(None, 6000, 3)]})

    e.run()
