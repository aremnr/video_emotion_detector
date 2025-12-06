from em_st_artifacts.utils import lib_settings, support_classes
from em_st_artifacts import emotional_math

class EmotionMathManager:
    def __init__(self, calibration_length=6, nwins_skip_after_artifact=10):
        self.mls = lib_settings.MathLibSetting(
            sampling_rate=250,
            process_win_freq=25,
            n_first_sec_skipped=4,
            fft_window=1000,
            bipolar_mode=True,
            channels_number=4,
            channel_for_analysis=0
        )
        self.ads = lib_settings.ArtifactDetectSetting(
            art_bord=110,
            allowed_percent_artpoints=70,
            raw_betap_limit=800_000,
            global_artwin_sec=4,
            num_wins_for_quality_avg=100,
            hamming_win_spectrum=True,
            hanning_win_spectrum=False,
            total_pow_border=100,
            spect_art_by_totalp=True
        )
        self.mss = lib_settings.MentalAndSpectralSetting(
            n_sec_for_averaging=2,
            n_sec_for_instant_estimation=4
        )
        self.math = emotional_math.EmotionalMath(self.mls, self.ads, self.mss)
        self.math.set_calibration_length(calibration_length)
        self.math.set_mental_estimation_mode(False)
        self.math.set_skip_wins_after_artifact(nwins_skip_after_artifact)
        self.math.set_zero_spect_waves(True, 0, 1, 1, 1, 0)
        self.math.set_spect_normalization_by_bands_width(True)

    def create_raw_channels(self, left_bipolar, right_bipolar):
        return support_classes.RawChannels(left_bipolar, right_bipolar)

    def push_bipolars(self, raw_channels):
        self.math.push_bipolars(raw_channels)

    def process_data_arr(self):
        self.math.process_data_arr()

    def calibration_finished(self):
        return self.math.calibration_finished()

    def get_calibration_percents(self):
        return self.math.get_calibration_percents()

    def read_mental_data_arr(self):
        return self.math.read_mental_data_arr()

    def read_spectral_data_percents_arr(self):
        return self.math.read_spectral_data_percents_arr()
