import logging

from scipy import signal

from chewbite_fusion.models.bloch import BlochModel
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features import feature_factories as ff

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_model_instance(variable_params):
    return BlochModel(input_size=variable_params['input_size'],
                      output_size=5,
                      n_epochs=1500,
                      batch_size=10)


def hamming_highpass_filter(signal_values):
    filter_values = signal.firwin(511,
                                  cutoff=0.1,
                                  fs=25,
                                  window="hamming",
                                  pass_zero="highpass")

    return signal.lfilter(filter_values, 1.0, signal_values)


@experiment()
def bloch2023_v1():
    """ Implementation based on Bloch 2023 (window size 0.3 s).
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          list(range(0, 9)),
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_RawIMUDataNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v1',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(30, 9)]})

    e.run()


@experiment()
def bloch2023_v2():
    """ Implementation based on Bloch 2023 (window size 0.5 s).
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          list(range(0, 9)),
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_RawIMUDataNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v2',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(50, 9)]})

    e.run()


@experiment()
def bloch2023_v3():
    """ Implementation based on Bloch 2023 (window size 1 s).
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          list(range(0, 9)),
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_RawIMUDataNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v3',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(100, 9)]})

    e.run()


@experiment()
def bloch2023_v4():
    """ Implementation based on Bloch 2023 (window size 0.3 s) and
        magnitudes from raw data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          [9, 10, 11],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_IMUMagnitudesNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v4',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(30, 3)]})

    e.run()


@experiment()
def bloch2023_v5():
    """ Implementation based on Bloch 2023 (window size 0.5 s) and
        magnitudes from raw data.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          [9, 10, 11],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_IMUMagnitudesNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v5',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(50, 3)]})

    e.run()


@experiment()
def bloch2023_v6():
    """ Implementation based on Bloch 2023 (window size 1.0 s) and
        magnitudes from raw data.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          [9, 10, 11],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_IMUMagnitudesNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v6',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(100, 3)]})

    e.run()


@experiment()
def bloch2023_v7():
    """ Implementation based on Bloch 2023 (window size 0.3 s) no mag.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          list(range(0, 9)),
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_RawIMUNoMagDataNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v7',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(30, 6)]})

    e.run()


@experiment()
def bloch2023_v8():
    """ Implementation based on Bloch 2023 (window size 0.5 s) no mag.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          list(range(0, 9)),
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_RawIMUNoMagDataNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v8',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(50, 6)]})

    e.run()


@experiment()
def bloch2023_v9():
    """ Implementation based on Bloch 2023 (window size 1 s) no mag.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          list(range(0, 9)),
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_RawIMUNoMagDataNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v9',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(100, 6)]})

    e.run()


@experiment()
def bloch2023_v10():
    """ Implementation based on Bloch 2023 (window size 0.3 s) and
        magnitudes from raw data, without magnetometer.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          [9, 10, 11],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_IMUMagnitudesNoMagNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v10',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(30, 2)]})

    e.run()


@experiment()
def bloch2023_v11():
    """ Implementation based on Bloch 2023 (window size 0.5 s) and
        magnitudes from raw data, without magnetometer.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          [9, 10, 11],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_IMUMagnitudesNoMagNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v11',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(50, 2)]})

    e.run()


@experiment()
def bloch2023_v12():
    """ Implementation based on Bloch 2023 (window size 1.0 s) and
        magnitudes from raw data, without magnetometer.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          [9, 10, 11],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_IMUMagnitudesNoMagNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v12',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(100, 2)]})

    e.run()


@experiment()
def bloch2023_v13():
    """ Implementation based on Bloch 2023 (window size 0.3 s) only accelerometer.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          [0, 1, 2],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_RawAccDataNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v13',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(30, 3)]})

    e.run()


@experiment()
def bloch2023_v14():
    """ Implementation based on Bloch 2023 (window size 0.5 s) only accelerometer.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          [0, 1, 2],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_RawAccDataNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v14',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(50, 3)]})

    e.run()


@experiment()
def bloch2023_v15():
    """ Implementation based on Bloch 2023 (window size 1.0 s) only accelerometer.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          [0, 1, 2],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_RawAccDataNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v15',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(100, 3)]})

    e.run()


@experiment()
def bloch2023_v16():
    """ Implementation based on Bloch 2023 (window size 0.3 s) and acc magnitudes.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          [9],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AccMagnitudesNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v16',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(30, 1)]})

    e.run()


@experiment()
def bloch2023_v17():
    """ Implementation based on Bloch 2023 (window size 0.5 s) and acc magnitudes.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          [9],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AccMagnitudesNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v17',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(50, 1)]})

    e.run()


@experiment()
def bloch2023_v18():
    """ Implementation based on Bloch 2023 (window size 1.0 s) and acc magnitudes.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000,
                filters=[(hamming_highpass_filter,
                          [9],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_AccMagnitudesNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v18',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(100, 1)]})

    e.run()


@experiment()
def bloch2023_v19():
    """ Implementation based on Bloch 2023 (window size 0.5 s) only accelerometer with
        sampling frequency 20 Hz.
    """
    window_width = 0.5
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                movement_sampling_frequency=20,
                filters=[(hamming_highpass_filter,
                          [0, 1, 2],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_RawAccDataNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v19',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(10, 3)]})

    e.run()


@experiment()
def bloch2023_v20():
    """ Implementation based on Bloch 2023 (window size 1.0 s) only accelerometer with
        sampling frequency 20 Hz.
    """
    window_width = 1.0
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=False,
                audio_sampling_frequency=6000,
                movement_sampling_frequency=20,
                filters=[(hamming_highpass_filter,
                          [0, 1, 2],
                          True)])

    e = Experiment(get_model_instance,
                   ff.FeatureFactory_RawAccDataNoSequences,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='bloch2023_v20',
                   manage_sequences=False,
                   use_raw_data=True,
                   model_parameters_grid={'input_size': [(20, 3)]})

    e.run()
