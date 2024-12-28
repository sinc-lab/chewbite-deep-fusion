import os
import logging
import pickle
from glob import glob
from datetime import datetime as dt
import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
import sed_eval
import dcase_util

from chewbite_fusion.data.utils import windows2events
from chewbite_fusion.experiments import settings
from chewbite_fusion.experiments.utils import set_random_init


logger = logging.getLogger('yaer')


class Experiment:
    ''' Base class to represent an experiment using audio and movement signals. '''
    def __init__(self,
                 model_factory,
                 features_factory,
                 X,
                 y,
                 window_width,
                 window_overlap,
                 name,
                 audio_sampling_frequency=8000,
                 movement_sampling_frequency=100,
                 no_event_class='no-event',
                 manage_sequences=False,
                 model_parameters_grid={},
                 use_raw_data=False,
                 quantization=None,
                 data_augmentation=False):
        self.timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        self.model_factory = model_factory
        self.features_factory = features_factory
        self.X = X
        self.y = y
        self.window_width = window_width
        self.window_overlap = window_overlap
        self.name = name
        self.audio_sampling_frequency = audio_sampling_frequency
        self.movement_sampling_frequency = movement_sampling_frequency
        self.no_event_class = no_event_class
        self.manage_sequences = manage_sequences
        self.model_parameters_grid = model_parameters_grid
        self.use_raw_data = use_raw_data
        self.train_validation_segments = []
        self.quantization = quantization
        self.data_augmentation = data_augmentation

        # Create path for experiment if needed.
        self.path = os.path.join(settings.experiments_path, name, self.timestamp)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # Add logger handlers (file and system stdout).
        logger.handlers = []
        fileHandler = logging.FileHandler(f"{self.path}/experiment.log")
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        # Set random init.
        set_random_init()

    def run(self):
        ''' Run the experiment and dump relevant information. '''
        self.X = self.X['zavalla2022']
        self.y = self.y['zavalla2022']

        # Segment assigment to each fold. This was created using random
        # sampling with stratified separation of rumination segments.
        folds = {
            '1': [45, 3, 23, 2, 17],
            '2': [20, 42, 21, 1, 39],
            '3': [28, 22, 33, 51, 55],
            '4': [10, 40, 14, 41, 19],
            '5': [47, 24, 7, 18]
        }

        for i in folds.values():
            self.train_validation_segments.extend(i)

        hash_method_instance = hashlib.new('sha256')
        params_results = {}
        full_grid = list(ParameterGrid(self.model_parameters_grid))

        if len(full_grid) > 1:
            for params_combination in full_grid:
                if params_combination != {}:
                    logger.info('Running folds for parameters combination: %s.',
                                params_combination)
                else:
                    logger.info('Running folds without grid search.')

                # Create parameters hash in order to compare results.
                hash_method_instance.update(str(params_combination).encode())
                params_combination_hash = hash_method_instance.hexdigest()

                params_combination_result = self.execute_kfoldcv(
                    folds=folds,
                    is_grid_search=True,
                    parameters_combination=params_combination)

                # Store result and params dict to be used if selected.
                params_results[params_combination_hash] = (params_combination_result,
                                                           params_combination)

            best_params_combination = max(params_results.values(), key=lambda i: i[0])[1]
            logger.info('-' * 25)
            logger.info('>>> All params combination values: %s <<<', str(params_results))
            logger.info('-' * 25)
            logger.info('>>> Best params combination: %s <<<', best_params_combination)
        else:
            logger.info('-' * 25)
            logger.info('>>> Skipping grid search! No params dict provided. <<<')
            best_params_combination = full_grid[0]

        self.execute_kfoldcv(
            folds=folds,
            is_grid_search=False,
            parameters_combination=best_params_combination)

    def execute_kfoldcv(self,
                        folds,
                        is_grid_search,
                        parameters_combination):
        ''' Execute a k-fold cross validation using a specific set of parameters. '''
        signal_predictions = {}

        for ix_fold, fold in folds.items():
            logger.info('Running fold number %s.', ix_fold)

            test_fold_keys = [k for k in self.X.keys() if int(k.split('_')[1]) in fold]
            train_fold_keys = [k for k in self.X.keys() if int(k.split('_')[1]) not in fold]
            train_fold_keys = [k for k in train_fold_keys if \
                int(k.split('_')[1]) in self.train_validation_segments]

            logger.info('Train fold keys: %s.', str(train_fold_keys))
            X_train = []
            y_train = []
            for train_signal_key in train_fold_keys:
                if self.manage_sequences:
                    X_train.append(self.X[train_signal_key])
                    y_train.append(self.y[train_signal_key])
                else:
                    X_train.extend(self.X[train_signal_key])
                    y_train.extend(self.y[train_signal_key])

            if self.data_augmentation:
                from augly.audio import functional
                # Compute classes distribution.
                all_y = []
                n_labels = 0
                for i_file in range(len(X_train)):
                    for i_window in range(len(X_train[i_file])):
                        if y_train[i_file][i_window] != 'no-event':
                            all_y.append(y_train[i_file][i_window])
                            n_labels += 1
                unique, counts = np.unique(all_y, return_counts=True)
                classes_probs = dict(zip(unique, counts / n_labels))

                # Create a copy of all training samples.
                import copy
                X_augmented = copy.deepcopy(X_train)
                y_augmented = copy.deepcopy(y_train)

                for i_file in range(len(X_train)):
                    during_event = False
                    discard_event = False
                    for i_window in range(len(X_train[i_file])):
                        window_label = y_train[i_file][i_window]
                        
                        if window_label == 'no-event':
                            during_event = False
                            discard_event = False
                        elif not during_event and window_label not in ['no-event',
                                                                       'bite',
                                                                       'rumination-chew']:
                            during_event = True
                            # If the windows correspond to a selected event to discard
                            # from a majority class, select it to make zero values and 'no-event'.
                            if np.random.rand() <= classes_probs[window_label] * 2:
                                discard_event = True

                        if during_event and discard_event:
                            for i_channel in range(len(X_train[i_file][i_window])):
                                window_len = len(X_augmented[i_file][i_window][i_channel])
                                X_augmented[i_file][i_window][i_channel] = np.zeros(window_len)
                                y_augmented[i_file][i_window] = 'no-event'
                        else:
                            for i_channel in range(len(X_train[i_file][i_window])):
                                if i_channel == 0:
                                    sample_rate = 6000
                                else:
                                    sample_rate = 100

                                window = X_augmented[i_file][i_window][i_channel]
                                X_augmented[i_file][i_window][i_channel] = \
                                    functional.add_background_noise(window,
                                                                    sample_rate,
                                                                    snr_level_db=20)[0]
                logger.info('Applying data augmentation !')
                logger.info(len(X_train))
                X_train.extend(X_augmented)
                y_train.extend(y_augmented)
                logger.info(len(X_train))

            # Create label encoder and fit with unique values.
            self.target_encoder = LabelEncoder()

            unique_labels = np.unique(np.hstack(y_train))
            self.target_encoder.fit(unique_labels)
            if self.manage_sequences:
                y_train_enc = []
                for file_labels in y_train:
                    y_train_enc.append(self.target_encoder.transform(file_labels))
            else:
                y_train_enc = self.target_encoder.transform(y_train)

            model_instance = self.model_factory(parameters_combination)
            self.model = model_instance
            self.set_model_output_path(ix_fold, is_grid_search)

            # Fit model and get predictions.
            funnel = Funnel(self.features_factory,
                            model_instance,
                            self.audio_sampling_frequency,
                            self.movement_sampling_frequency,
                            self.use_raw_data)
            funnel.fit(X_train, y_train_enc)

            if self.quantization:
                for ix_layer, layer in enumerate(funnel.model.model.layers):
                    w = layer.get_weights()
                    w = [i.astype(self.quantization) for i in w]
                    funnel.model.model.layers[ix_layer].set_weights(w)
                logger.info('quantization applied correctly !', str(self.quantization))

            for test_signal_key in test_fold_keys:
                if self.manage_sequences:
                    X_test = [self.X[test_signal_key]]
                else:
                    X_test = self.X[test_signal_key]

                y_signal_pred = funnel.predict(X_test)

                if self.manage_sequences:
                    y_signal_pred = y_signal_pred[0]

                y_signal_pred_labels = self.target_encoder.inverse_transform(y_signal_pred)

                y_test = self.y[test_signal_key]
                signal_predictions[test_signal_key] = [y_test, y_signal_pred_labels]

        logger.info('-' * 25)
        logger.info('Fold iterations finished !. Starting evaluation phase.')

        # Save predictions.
        self.save_predictions(signal_predictions)

        unique_labels = np.concatenate([self.y[k] for k in self.y.keys()])
        unique_labels = list(set(unique_labels))

        if self.no_event_class in unique_labels:
            unique_labels.remove(self.no_event_class)

        if is_grid_search:
            fold_metrics = self.evaluate(unique_labels=unique_labels,
                                         folds=folds,
                                         verbose=False)
        else:
            # Log general information about experiment result.
            logger.info('-' * 50)
            logger.info('***** Classification results *****')
            fold_metrics = self.evaluate(unique_labels=unique_labels,
                                         folds=folds,
                                         verbose=True)

        return fold_metrics

    def save_predictions(self,
                         fold_labels_predictions):
        ''' Save predictions to disk processing windows and labels. '''
        df = pd.DataFrame(columns=['segment', 'y_true', 'y_pred'])

        for segment in fold_labels_predictions.keys():
            y_true = fold_labels_predictions[segment][0]
            y_pred = fold_labels_predictions[segment][1]

            _df = pd.DataFrame({
                'y_true': y_true,
                'y_pred': y_pred
            })
            _df['segment'] = segment
            df = pd.concat([df, _df])

            # Transform windows to events and save.
            df_labels = windows2events(y_true,
                                       self.window_width,
                                       self.window_overlap)
            # Remove no-event class.
            df_labels = df_labels[df_labels.label != self.no_event_class]
            df_labels.to_csv(os.path.join(self.path, segment + '_true.txt'),
                             sep='\t',
                             header=False,
                             index=False)

            df_predictions = windows2events(y_pred,
                                            self.window_width,
                                            self.window_overlap)
            # Remove no-event class.
            df_predictions = df_predictions[df_predictions.label != self.no_event_class]
            df_predictions.to_csv(os.path.join(self.path, segment + '_pred.txt'),
                                  sep='\t',
                                  header=False,
                                  index=False)

        df.to_csv(os.path.join(self.path, 'fold_labels_and_predictions.csv'))

    def evaluate(self, unique_labels, folds, verbose=True):
        target_files = glob(os.path.join(self.path, 'segment_*_true.txt'))
        final_metric = 'f_measure'

        # Dictionary used to save selected metric per fold.
        fold_metrics_detail = {}
        fold_metrics = []

        for ix_fold, fold in folds.items():
            file_list = []
            fold_files = [f for f
                          in target_files if int(os.path.basename(f).split('_')[1]) in fold]
            for file in fold_files:
                pred_file = file.replace('true', 'pred')
                file_list.append({
                    'reference_file': file,
                    'estimated_file': pred_file
                })

            data = []

            # Get used event labels
            all_data = dcase_util.containers.MetaDataContainer()
            for file_pair in file_list:
                reference_event_list = sed_eval.io.load_event_list(
                    filename=file_pair['reference_file']
                )
                estimated_event_list = sed_eval.io.load_event_list(
                    filename=file_pair['estimated_file']
                )

                data.append({'reference_event_list': reference_event_list,
                             'estimated_event_list': estimated_event_list})

                all_data += reference_event_list

            # Create metrics classes, define parameters
            segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
                event_label_list=unique_labels,
                time_resolution=settings.segment_width_value
            )

            event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
                event_label_list=unique_labels,
                t_collar=settings.collar_value
            )

            # Go through files
            for file_pair in data:
                segment_based_metrics.evaluate(
                    reference_event_list=file_pair['reference_event_list'],
                    estimated_event_list=file_pair['estimated_event_list']
                )

                event_based_metrics.evaluate(
                    reference_event_list=file_pair['reference_event_list'],
                    estimated_event_list=file_pair['estimated_event_list']
                )

            # Dump metrics objects in order to facilitate comparision and reports generation.
            metrics = {
                'segment_based_metrics': segment_based_metrics,
                'event_based_metrics': event_based_metrics
            }

            dump_file_name = f'experiment_metrics_fold_{ix_fold}.pkl'
            with open(os.path.join(self.path, dump_file_name), 'wb') as handle:
                pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

            segment_metrics = segment_based_metrics.results_overall_metrics()
            event_metrics = event_based_metrics.results_overall_metrics()

            if verbose:
                logger.info('### Segment based metrics (fold %s) ###', ix_fold)
                logger.info(segment_based_metrics)
                logger.info('')
                logger.info('### Event based metrics (fold %s) ###', ix_fold)
                logger.info(event_based_metrics)
                logger.info('-' * 20)

            fold_metrics_detail[ix_fold] = {
                'event_score': event_metrics[final_metric],
                'segment_score': segment_metrics[final_metric]
            }
            fold_metrics.append(event_metrics[final_metric][final_metric])

        dump_file_name = 'experiment_overall_metrics.pkl'
        with open(os.path.join(self.path, dump_file_name), 'wb') as handle:
            pickle.dump(fold_metrics_detail, handle, protocol=pickle.HIGHEST_PROTOCOL)

        folds_mean = np.round(np.mean(fold_metrics), 6)
        folds_std = np.round(np.std(fold_metrics), 6)

        if verbose:
            logger.info('### Event based overall metrics ###')
            logger.info('F1 score (micro) mean for events: %s', str(folds_mean))
            logger.info('F1 score (micro) standard deviation for events: %s', str(folds_std))
            logger.info('-' * 20)

        return folds_mean

    def set_model_output_path(self, n_fold, is_grid_search=False):
        output_logs_path = os.path.join(self.path, f'logs_fold_{n_fold}')
        output_model_checkpoint_path = os.path.join(self.path, f'model_checkpoints_fold_{n_fold}')

        # Skip validation of directory existance during grid search.
        if not is_grid_search:
            # Check if paths already exists
            if os.path.exists(output_logs_path):
                assert not os.path.exists(output_logs_path), 'Model output logs path already exists!'

            if os.path.exists(output_model_checkpoint_path):
                assert not os.path.exists(output_model_checkpoint_path), \
                    'Model output checkpoints path already exists!'

        # If not, create and save into model instance.
        os.makedirs(output_logs_path)
        self.model.output_logs_path = output_logs_path

        os.makedirs(output_model_checkpoint_path)
        self.model.output_path_model_checkpoints = output_model_checkpoint_path


class Funnel:
    ''' A similar interface to sklearn Pipeline, but transformations are applied in parallel. '''
    def __init__(self,
                 features_factory,
                 model_instance,
                 audio_sampling_frequency,
                 movement_sampling_frequency,
                 use_raw_data=False):
        self.features = features_factory(
            audio_sampling_frequency,
            movement_sampling_frequency).features
        self.model = model_instance
        self.use_raw_data = use_raw_data

    def fit(self, X, y):
        # Fit features and transform data.
        X_features = []

        for feature in self.features:
            logger.info(f'Processing the feature {feature.feature.__class__.__name__}.')
            X_features.append(feature.fit_transform(X, y))

        if not self.use_raw_data:
            X_features = np.concatenate(X_features, axis=1)

        # Fit model.
        logger.info('Training model ...')
        self.model.fit(X_features, y)

    def predict(self, X):
        # Transform data using previously fitted features.
        X_features = []

        for feature in self.features:
            X_features.append(feature.transform(X))

        if not self.use_raw_data:
            X_features = np.concatenate(X_features, axis=1)

        # Get model predictions with transformed data.
        return self.model.predict(X_features)
