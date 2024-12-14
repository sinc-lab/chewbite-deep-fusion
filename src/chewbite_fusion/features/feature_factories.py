from sklearn.preprocessing import StandardScaler

from chewbite_fusion.features.base import BaseFeatureBuilder
from chewbite_fusion.features import imu_statistical_based_features as sbf
from chewbite_fusion.features import imu_statistical_based_features_seq as sbfs
from chewbite_fusion.features import audio_spectral_based_features as aspf
from chewbite_fusion.features import audio_statistical_based_features as asf
from chewbite_fusion.features import audio_cbia_based_features as cbia
from chewbite_fusion.features import audio_cbia_based_features_seq as cbia_seq
from chewbite_fusion.features import audio_raw_data as ard
from chewbite_fusion.features import imu_raw_data as ird
from chewbite_fusion.features import imu_raw_data_no_sequences as irdns
from chewbite_fusion.features import imu_diff as id


class BaseFeatureFactory():
    def __init__(self, features,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        self.features = []
        for feature in features:
            self.features.append(BaseFeatureBuilder(
                feature,
                audio_sampling_frequency,
                movement_sampling_frequency,
                StandardScaler
            ))


class BaseFeatureFactoryNoPreprocessing():
    def __init__(self, features,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        self.features = []
        for feature in features:
            self.features.append(BaseFeatureBuilder(
                feature,
                audio_sampling_frequency,
                movement_sampling_frequency,
                None
            ))


class FeatureFactory_v1(BaseFeatureFactory):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            sbf.MovementSignalAccAverage,
            sbf.MovementSignalGyrAverage,
            sbf.MovementSignalMagAverage,
            sbf.MovementSignalAccStandardDeviation,
            sbf.MovementSignalGyrStandardDeviation,
            sbf.MovementSignalMagStandardDeviation,
            sbf.MovementSignalAccMax,
            sbf.MovementSignalGyrMax,
            sbf.MovementSignalMagMax,
            sbf.MovementSignalAccMin,
            sbf.MovementSignalGyrMin,
            sbf.MovementSignalMagMin
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_Alvarenga2019(BaseFeatureFactory):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            sbf.MovementSignalAccAverage,
            sbf.MovementSignalAccStandardDeviation,
            sbf.MovementSignalAccMin,
            sbf.MovementSignalAccMax,
            sbf.MovementSignalAccAreaStats,
            sbf.MovementSignalAccMagnitudeStats,
            sbf.MovementSignalAccVariation,
            sbf.MovementSignalAccEnergyStats,
            sbf.MovementSignalAccEntropyStats,
            sbf.MovementSignalAccPitchStats,
            sbf.MovementSignalAccRollStats,
            sbf.MovementSignalAccInclinationStats
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_v2(BaseFeatureFactory):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            asf.AudioSignalAverage,
            asf.AudioSignalEnergy,
            asf.AudioSignalKurtosis,
            asf.AudioSignalMax,
            asf.AudioSignalMin,
            asf.AudioSignalStandardDeviation,
            asf.AudioSignalSum,
            asf.AudioSignalZeroCrossing,
            aspf.AudioSpectralCentroid
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_v3(BaseFeatureFactory):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            sbf.MovementSignalAccAverage,
            sbf.MovementSignalAccAreaStats,
            sbf.MovementSignalAccEnergyStats,
            sbf.MovementSignalAccEntropyStats,
            sbf.MovementSignalAccInclinationStats,
            sbf.MovementSignalAccMagnitudeStats,
            sbf.MovementSignalAccMax,
            sbf.MovementSignalAccMin,
            sbf.MovementSignalGyrAverage,
            sbf.MovementSignalGyrMax,
            sbf.MovementSignalGyrMin,
            sbf.MovementSignalGyrStandardDeviation,
            asf.AudioSignalAverage,
            asf.AudioSignalEnergy,
            asf.AudioSignalKurtosis,
            asf.AudioSignalMax,
            asf.AudioSignalMin,
            asf.AudioSignalStandardDeviation,
            asf.AudioSignalSum,
            asf.AudioSignalZeroCrossing,
            aspf.AudioSpectralCentroid
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_v4(BaseFeatureFactory):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            asf.AudioSignalAverage,
            asf.AudioSignalEnergy,
            asf.AudioSignalKurtosis,
            asf.AudioSignalMax,
            asf.AudioSignalMin,
            asf.AudioSignalStandardDeviation,
            asf.AudioSignalSum,
            asf.AudioSignalZeroCrossing,
            aspf.AudioSpectralCentroid,
            cbia.CBIAFeatures
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_v5(BaseFeatureFactory):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            sbf.MovementSignalAccAreaStats,
            sbf.MovementSignalGyrAreaStats,
            sbf.MovementSignalAccMagnitudeStats,
            sbf.MovementSignalGyrMagnitudeStats,
            sbf.MovementSignalAccVariation,
            sbf.MovementSignalGyrVariation,
            sbf.MovementSignalAccEnergyStats,
            sbf.MovementSignalGyrEnergyStats,
            sbf.MovementSignalAccEntropyStats,
            sbf.MovementSignalGyrEntropyStats,
            sbf.MovementSignalAccPitchStats,
            sbf.MovementSignalAccRollStats,
            sbf.MovementSignalAccInclinationStats,
            cbia.CBIAFeatures
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_RawAudioData(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            ard.AudioRawData
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_RawIMUData(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            ird.AccXRawData,
            ird.AccYRawData,
            ird.AccZRawData,
            ird.GyrXRawData,
            ird.GyrYRawData,
            ird.GyrZRawData,
            ird.MagXRawData,
            ird.MagYRawData,
            ird.MagZRawData
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_RawIMUDataNoSequences(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            irdns.AccXRawData,
            irdns.AccYRawData,
            irdns.AccZRawData,
            irdns.GyrXRawData,
            irdns.GyrYRawData,
            irdns.GyrZRawData,
            irdns.MagXRawData,
            irdns.MagYRawData,
            irdns.MagZRawData
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_RawIMUNoMagDataNoSequences(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            irdns.AccXRawData,
            irdns.AccYRawData,
            irdns.AccZRawData,
            irdns.GyrXRawData,
            irdns.GyrYRawData,
            irdns.GyrZRawData
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_RawAccDataNoSequences(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            irdns.AccXRawData,
            irdns.AccYRawData,
            irdns.AccZRawData
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_IMUMagnitudesNoSequences(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            irdns.AccMagnitudeVector,
            irdns.GyrMagnitudeVector,
            irdns.MagMagnitudeVector
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_IMUMagnitudesNoMagNoSequences(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            irdns.AccMagnitudeVector,
            irdns.GyrMagnitudeVector
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_AccMagnitudesNoSequences(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            irdns.AccMagnitudeVector
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_AllRawData(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            ard.AudioRawData,
            ird.AccXRawData,
            ird.AccYRawData,
            ird.AccZRawData,
            ird.GyrXRawData,
            ird.GyrYRawData,
            ird.GyrZRawData,
            ird.MagXRawData,
            ird.MagYRawData,
            ird.MagZRawData
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_AudioAccGyrRawData(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            ard.AudioRawData,
            ird.AccXRawData,
            ird.AccYRawData,
            ird.AccZRawData,
            ird.GyrXRawData,
            ird.GyrYRawData,
            ird.GyrZRawData
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_AudioAccGyrVectorsRawData(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            ard.AudioRawData,
            ird.AccMagnitudeVector,
            ird.GyrMagnitudeVector
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_AudioAccGyrDiffVectorsRawData(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            ard.AudioRawData,
            id.AccMagnitudeDiffVector,
            id.GyrMagnitudeDiffVector
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_DecisionLevelMixData(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            ard.AudioRawData,
            sbfs.MovementSignalAccAverage,
            sbfs.MovementSignalAccStandardDeviation,
            sbfs.MovementSignalAccMin,
            sbfs.MovementSignalAccMax,
            sbfs.MovementSignalAccAreaStats,
            sbfs.MovementSignalAccMagnitudeStats,
            sbfs.MovementSignalAccVariation,
            sbfs.MovementSignalAccEnergyStats,
            sbfs.MovementSignalAccEntropyStats,
            sbfs.MovementSignalAccPitchStats,
            sbfs.MovementSignalAccRollStats,
            sbfs.MovementSignalAccInclinationStats
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_DecisionLevelMixData_v2(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            ard.AudioRawData,
            cbia_seq.CBIAFeatures,
            ird.AccXRawData,
            ird.AccYRawData,
            ird.AccZRawData,
            ird.GyrXRawData,
            ird.GyrYRawData,
            ird.GyrZRawData,
            ird.MagXRawData,
            ird.MagYRawData,
            ird.MagZRawData,
            sbfs.MovementSignalAccAverage,
            sbfs.MovementSignalAccStandardDeviation,
            sbfs.MovementSignalAccMin,
            sbfs.MovementSignalAccMax,
            sbfs.MovementSignalAccAreaStats,
            sbfs.MovementSignalAccMagnitudeStats,
            sbfs.MovementSignalAccVariation,
            sbfs.MovementSignalAccEnergyStats,
            sbfs.MovementSignalAccEntropyStats,
            sbfs.MovementSignalAccPitchStats,
            sbfs.MovementSignalAccRollStats,
            sbfs.MovementSignalAccInclinationStats
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)


class FeatureFactory_AudioAccRawData(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            ard.AudioRawData,
            ird.AccXRawData,
            ird.AccYRawData,
            ird.AccZRawData
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)
