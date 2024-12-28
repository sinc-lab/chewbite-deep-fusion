import logging

from chewbite_fusion.models import ablation_models as am
from chewbite_fusion.models import deep_fusion_feature_level as dffl
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment, PredictionExperiment
from chewbite_fusion.features import feature_factories as ff

from yaer.base import experiment


logger = logging.getLogger('yaer')


def get_model_A_instance(variable_params):
    return am.DeepFusionAblationA(
        input_size_audio=variable_params['input_size_audio'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True)

def get_model_B_instance(variable_params):
    return am.DeepFusionAblationB(
        input_size_acc=variable_params['input_size_acc'],
        input_size_gyr=variable_params['input_size_gyr'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True)

def get_model_B_a_instance(variable_params):
    return am.DeepFusionAblationB_a(
        input_size_acc=variable_params['input_size_acc'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True)

def get_model_B_b_instance(variable_params):
    return am.DeepFusionAblationB(
        input_size_gyr=variable_params['input_size_gyr'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True)

def get_model_C_instance(variable_params):
    return am.DeepFusionAblationC(
        input_size_acc=variable_params['input_size_acc'],
        input_size_gyr=variable_params['input_size_gyr'],
        input_size_audio=variable_params['input_size_sound'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True)

def get_model_C_a_instance(variable_params):
    return am.DeepFusionAblationC(
        input_size_acc=variable_params['input_size_acc'],
        input_size_audio=variable_params['input_size_sound'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True)

def get_model_C_b_instance(variable_params):
    return am.DeepFusionAblationC(
        input_size_gyr=variable_params['input_size_gyr'],
        input_size_audio=variable_params['input_size_sound'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True)

def get_model_D_instance(variable_params):
    return am.DeepFusionAblationD(
        input_size_acc=variable_params['input_size_acc'],
        input_size_gyr=variable_params['input_size_gyr'],
        input_size_audio=variable_params['input_size_sound'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True)

def get_model_E_instance(variable_params):
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


@experiment()
def ablation_model_A_validation():
    """ Experiment with sound head on validation data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_A_instance,
                   ff.FeatureFactory_AllRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='ablation_model_A_validation',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_audio': [(None, 1800, 1)]})

    e.run()


@experiment()
def ablation_model_B_validation():
    """ Experiment with IMU heads on validation data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_B_instance,
                   ff.FeatureFactory_AllRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='ablation_model_B_validation',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 30, 3)],
                                          'input_size_gyr': [(None, 30, 3)]})

    e.run()
    
@experiment()
def ablation_model_B_a_validation():
    """ Experiment with acc head on validation data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_B_a_instance,
                   ff.FeatureFactory_AllRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='ablation_model_B_a_validation',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 30, 3)]})

    e.run()


@experiment()
def ablation_model_B_b_validation():
    """ Experiment with gyr head on validation data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_B_b_instance,
                   ff.FeatureFactory_AllRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='ablation_model_B_b_validation',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_gyr': [(None, 30, 3)]})

    e.run()


@experiment()
def ablation_model_C_validation():
    """ Experiment without RNN on validation data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_C_instance,
                   ff.FeatureFactory_AllRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='ablation_model_C_validation',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 30, 3)],
                                          'input_size_gyr': [(None, 30, 3)],
                                          'input_size_sound': [(None, 1800, 1)]})

    e.run()


@experiment()
def ablation_model_D_validation():
    """ Experiment without dense layers on validation data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = Experiment(get_model_D_instance,
                   ff.FeatureFactory_AllRawData,
                   X, y,
                   window_width=window_width,
                   window_overlap=window_overlap,
                   name='ablation_model_D_validation',
                   manage_sequences=True,
                   use_raw_data=True,
                   model_parameters_grid={'input_size_acc': [(None, 30, 3)],
                                          'input_size_gyr': [(None, 30, 3)],
                                          'input_size_sound': [(None, 1800, 1)]})

    e.run()


@experiment()
def ablation_model_A_test():
    """ Experiment with sound head on test data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = PredictionExperiment(
        get_model_A_instance,
        ff.FeatureFactory_AllRawData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='ablation_model_A_test',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_audio': [(None, 1800, 1)]}
    )

    e.run()


@experiment()
def ablation_model_B_test():
    """ Experiment with IMU heads on test data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = PredictionExperiment(
        get_model_B_instance,
        ff.FeatureFactory_AllRawData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='ablation_model_B_test',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_acc': [(None, 30, 3)],
                               'input_size_gyr': [(None, 30, 3)]}
    )

    e.run()


@experiment()
def ablation_model_B_a_test():
    """ Experiment with acc head on test data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = PredictionExperiment(
        get_model_B_a_instance,
        ff.FeatureFactory_AllRawData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='ablation_model_B_a_test',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_acc': [(None, 30, 3)]}
    )

    e.run()


@experiment()
def ablation_model_B_b_test():
    """ Experiment with gyr head on test data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = PredictionExperiment(
        get_model_B_b_instance,
        ff.FeatureFactory_AllRawData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='ablation_model_B_b_test',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_gyr': [(None, 30, 3)]}
    )

    e.run()


@experiment()
def ablation_model_C_test():
    """ Experiment without RNN on test data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = PredictionExperiment(
        get_model_C_instance,
        ff.FeatureFactory_AllRawData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='ablation_model_C_test',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_acc': [(None, 30, 3)],
                               'input_size_gyr': [(None, 30, 3)],
                               'input_size_sound': [(None, 1800, 1)]})

    e.run()


@experiment()
def ablation_model_D_test():
    """ Experiment without dense layers on test data.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = PredictionExperiment(
        get_model_D_instance,
        ff.FeatureFactory_AllRawData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='ablation_model_D_test',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_acc': [(None, 30, 3)],
                               'input_size_gyr': [(None, 30, 3)],
                               'input_size_sound': [(None, 1800, 1)]})

    e.run()


@experiment()
def ablation_model_E_test():
    """ Experiment proposed model on test dataset.
    """
    window_width = 0.3
    window_overlap = 0.5
    X, y = main(window_width=window_width,
                window_overlap=window_overlap,
                include_movement_magnitudes=True,
                audio_sampling_frequency=6000)

    e = PredictionExperiment(
        get_model_E_instance,
        ff.FeatureFactory_AllRawData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='ablation_model_E_test',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_acc': [(None, 30, 3)],
                               'input_size_gyr': [(None, 30, 3)],
                               'input_size_audio': [(None, 1800, 1)]})

    e.run()
