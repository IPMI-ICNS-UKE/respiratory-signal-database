"""
Please scroll down to if __name__ == "__main__":
"""

from __future__ import annotations

import logging
from pathlib import Path
from random import randrange

import matplotlib.pyplot as plt
from torch import Tensor
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from resp_db.client import RpmDatabaseClient
from resp_db.common_types import PathLike
from resp_db.logger import LoggerMixin, init_fancy_logging
from resp_db.time_series_utils import add_white_noise_to_signal, fourier_smoothing


class RpmSignals(Dataset, LoggerMixin):
    """This is a dummy dataset and dataloader, which might be helpful in your
    project.

    Respiratory signals are queried from the database. Then, each signal
    is preprocessed on the fly (i.e. in the __getitem__ function). Note
    that, signals have varying signal length, which complicates batch
    learning. Either use a batch size of one or crop signals to the same
    length (see length_signal_snippet_s parameter). Further, depending on your purposes, you might want to
    smooth or add noise to the signals as well. Check out our suggest
    functions in the __getitem__ function below.
    """

    def __init__(
            self,
            db_root: PathLike,
            mode: str = "train",
            fourier_smooting_hz: int | None = 1,
            white_noise_db: int | None = 30,
            length_signal_snippet_s: int | None = None,
            min_length_s: int = 59,
            sampling_rate_hz: int = 25,
    ):
        super().__init__()
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"{mode=} not supported in sql-driven database {db_root}")
        if length_signal_snippet_s and length_signal_snippet_s > min_length_s:
            raise ValueError(
                f"{length_signal_snippet_s=} is greater than {min_length_s=}. \n"
                f"Hence, a random signal snippet of length {length_signal_snippet_s} cannot be extracted!"
            )
        self.mode = mode
        self.db_root = Path(db_root)
        self.white_noise_db = white_noise_db
        self.fourier_cutoff = fourier_smooting_hz
        self.sampling_rate = sampling_rate_hz
        self.length_signal_snippet_s = length_signal_snippet_s
        client = RpmDatabaseClient(db_filepath=db_root)
        with client:
            query = client.get_signals_of_dl_dataset(
                dl_dataset=mode, project="short-term-prediction"
            )
        len_unfiltered_query = len(query)

        self.query = [
            signal
            for signal in query
            if client.preprocess_signal(signal.df_signal).time.max() > min_length_s
        ]
        self.logger.info(
            f"{mode.upper()} dataset was successfully loaded. It contains {len(self.query)} signals."
        )
        self.logger.debug(
            f"Entire {mode.upper()} dataset contains {len_unfiltered_query} signals. "
            f"However, {len_unfiltered_query - len(self.query)} signals were sorted out "
            f"since they are shorter than {min_length_s=} s."
        )

    def __len__(self):
        return len(self.query)

    def __getitem__(self, index: int) -> tuple[str, Tensor, Tensor, Tensor]:
        signal = self.query[index]
        research_number, modality, fraction = (
            signal.research_number,
            signal.modality,
            signal.fraction,
        )
        name = f"{research_number}_{modality}_{fraction}"
        df_signal = RpmDatabaseClient.preprocess_signal(
            df_signal=signal.df_signal,
            sampling_rate=self.sampling_rate,
            only_beam_on=True,
            remove_offset=True,
        )
        # df.columns = ["time", "amplitude"]; [time] : second, [amplitude]: centimetres

        # select random snippet of length length_signal_snippet_s to allow signal stacking
        # (same length each signal -> batching)
        if self.length_signal_snippet_s:
            start_signal_index, end_signal_index = self.select_random_subset(
                time_series_len=len(df_signal),
                signal_length_s=self.length_signal_snippet_s,
                samples_per_second=self.sampling_rate,
            )
            df_signal = df_signal.iloc[start_signal_index:end_signal_index]

        # if you aim to perform smoothing
        time_series_smooth = fourier_smoothing(
            time_series=df_signal.amplitude.values,
            freq_threshold_hz=self.fourier_cutoff,
            sampling_rate=self.sampling_rate,
            return_spectrum=False,
        )
        self.logger.debug(
            f"{name} was fourier smoothed; cutoff {self.fourier_cutoff} Hz."
        )
        # if you aim to add noise to the signal
        time_series_noisy = add_white_noise_to_signal(
            target_snr_db=self.white_noise_db, signal=time_series_smooth
        )
        self.logger.debug(
            f"White noise of  {self.white_noise_db} dB was added to {name}"
        )

        return name, torch.from_numpy(df_signal.time.values.astype(np.float32)), torch.from_numpy(
            time_series_smooth.astype(np.float32)), torch.from_numpy(time_series_noisy.astype(np.float32))

    @staticmethod
    def select_random_subset(
            time_series_len: int,
            signal_length_s: int,
            samples_per_second: int,
    ) -> tuple[int, int]:
        num_points = samples_per_second * signal_length_s
        if time_series_len < num_points:
            raise ValueError("Time series shorter than desired signal")
        if time_series_len - 1 - num_points <= 0:
            raise ValueError(
                "Cannot pick a random snippet cause time series is to short"
            )
        start_signal_index = randrange(0, time_series_len - 1 - num_points)
        end_signal_index = start_signal_index + num_points
        return start_signal_index, end_signal_index


if __name__ == "__main__":
    # if you aim to use this template, you have install torch first:
    # Either do: pip3 install torch;  Or: check out official website https://pytorch.org/get-started/locally/
    init_fancy_logging()
    logging.getLogger("resp_db").setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    database_root = Path(".../open_access_rpm_signals_master.db")  # change to path of downloaded database
    batch_size = 128
    train_dataset = RpmSignals(
        db_root=database_root,
        mode="train",
        fourier_smooting_hz=1,
        white_noise_db=27,
        min_length_s=60,
        sampling_rate_hz=25,
        length_signal_snippet_s=50,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    for names, times, signals_smooth, signal_noisy in train_loader:
        logger.info(f"{times.shape}")
        logger.info(f"{signals_smooth.shape=}")
        logger.info(f"{signal_noisy.shape=}")
        fig, ax = plt.subplots(1, 1)
        # plot first curve of batch
        ax.plot(times[0], signal_noisy[0], label="noisy")
        ax.plot(times[0], signals_smooth[0], label="smooth")
        ax.legend(loc=1)
        ax.set_title(f"{names[0]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (cm)")
        plt.show()
        break

    # test_dataset = RpmSignals(db_root=database_filepath,
    #                           mode="test",
    #                           fourier_smooting_hz=1,
    #                           white_noise_db=27,
    #                           min_length_s=60,
    #                           sampling_rate=25
    #                           )
    #
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    # )
