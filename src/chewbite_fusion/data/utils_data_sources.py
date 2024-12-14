from collections import namedtuple
import os
import glob

from chewbite_fusion.data.settings import DATA_SOURCES_PATH


def list_datasets():
    """ Return a dictionary with available datasets. """

    Dataset = namedtuple("dataset",
                         ["str_id", "name", "folder", "audio_files_format",
                          "audio_sampling_frequency", "imu_sampling_frequency", "multimodal"])

    assert os.path.exists(DATA_SOURCES_PATH), f"Path {DATA_SOURCES_PATH} does not exist."

    datasets = {
        'zavalla2022': Dataset(
            str_id="jm2022",
            name="Jaw Movement 2022 Dataset with Smartphones used a recording device.",
            folder=os.path.join(DATA_SOURCES_PATH, "jm2022"),
            audio_files_format="wav",
            audio_sampling_frequency=22050,
            imu_sampling_frequency=100,
            multimodal=True
        )
    }

    return datasets


def get_files_in_dataset(dataset):
    """ Get list of files included in given dataset instance. """

    if dataset.audio_files_format == "wav":
        ext = "wav"

    dataset_files = []
    labels_file_list = sorted(glob.glob(os.path.join(dataset.folder, "*labels.txt")))

    for label_file in labels_file_list:
        audio_file = label_file.replace('_labels.txt', '.' + ext)

        files_group = [audio_file]
        if dataset.multimodal:
            for sensor in ['acc', 'gyr', 'mag']:
                for axis in ['x', 'y', 'z']:
                    file = label_file.replace('_labels.txt', f'_{sensor}_{axis}.txt')
                    files_group.append(file)
        files_group.append(label_file)

        # Check if files exist.
        for file in files_group:
            assert os.path.isfile(file), f'Could not find a specific file: {file}'

        dataset_files.append(tuple(files_group))

    return dataset_files
