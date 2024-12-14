import os
import glob
import random as python_random

import sed_eval
import dcase_util
import numpy as np
import tensorflow as tf

from chewbite_fusion.experiments import settings
from chewbite_fusion.experiments.settings import random_seed


def set_random_init():
    # Random seeds initialization in order to force results reproducibility.
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    python_random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def get_experiment_results(experiment_path, full=False):
    folds = {
        '1': [45, 3, 23, 2, 17],
        '2': [20, 42, 21, 1, 39],
        '3': [28, 22, 33, 51, 55],
        '4': [10, 40, 14, 41, 19],
        '5': [47, 24, 7, 18]
    }

    target_files = glob.glob(os.path.join(experiment_path, 'segment_*_true.txt'))

    fold_metrics = []

    unique_labels = ['grazing-chew', 'rumination-chew', 'bite', 'chewbite']

    for _, fold in folds.items():
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

        event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=unique_labels,
            t_collar=settings.collar_value
        )

        # Go through files
        for file_pair in data:
            event_based_metrics.evaluate(
                reference_event_list=file_pair['reference_event_list'],
                estimated_event_list=file_pair['estimated_event_list']
            )

        if full:
            fold_metrics.append(event_based_metrics)
        else:
            fold_metrics.append(event_based_metrics.results_overall_metrics())

    return fold_metrics
