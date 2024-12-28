import logging

from chewbite_fusion.models import deep_fusion_feature_level as dfflm
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features import feature_factories as ff

from yaer.base import experiment

from keras.layers.merging import average, maximum, multiply

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
        feature_scaling=True
        )


@experiment()
def deep_fusion_feature_level_m7_v7_augmentation():
    """ Experiment with fusion network (3 heads CNN) at feature level with window width 0.3s,
        using audio, accelerometer and gyroscope raw data plus data augmentation.
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
                   name='deep_fusion_feature_level_m7_v7_augmentation',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 30, 3)],
                                          'input_size_gyr': [(None, 30, 3)],
                                          'input_size_mag': [None],
                                          'input_size_audio': [(None, 1800, 1)]},
                   data_augmentation=True)

    e.run()
