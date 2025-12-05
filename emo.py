import math

from em_st_artifacts.utils.support_classes import RawChannelsArray, RawChannels

from em_st_artifacts.emotional_math import EmotionalMath

from em_st_artifacts.utils.lib_settings import MathLibSetting, ArtifactDetectSetting, \
    MentalAndSpectralSetting

mls = MathLibSetting(sampling_rate=250,
                                  process_win_freq=25,
										 
                                  n_first_sec_skipped=4,
                                  fft_window=1000,
                                  bipolar_mode=True,
                                  squared_spectrum=True,
                                  channels_number=4,
                                  channel_for_analysis=0)

ads = ArtifactDetectSetting(art_bord=110,
                                         allowed_percent_artpoints=70,
                                         raw_betap_limit=800_000,
                                         global_artwin_sec=4,
                                         num_wins_for_quality_avg=125,
                                         hamming_win_spectrum=True,
                                         hanning_win_spectrum=False,
                                         total_pow_border=400_000_000,
                                         spect_art_by_totalp=True)

mss = MentalAndSpectralSetting(n_sec_for_averaging=2,
                                            n_sec_for_instant_estimation=4)

math = EmotionalMath(mls, ads, mss)

# setting calibration length
calibration_length = 6
math.set_calibration_length(calibration_length)

# type of evaluation of instant mental levels
independent_mental_levels = False
math.set_mental_estimation_mode(independent_mental_levels)

# number of windows after the artifact with the previous actual value
nwins_skip_after_artifact = 10
math.set_skip_wins_after_artifact(nwins_skip_after_artifact)

# calculation of mental levels relative to calibration values
math.set_zero_spect_waves(True, 0, 1, 1, 1, 0)

# spectrum normalization by bandwidth
math.set_spect_normalization_by_bands_width(True)

math.start_calibration()
