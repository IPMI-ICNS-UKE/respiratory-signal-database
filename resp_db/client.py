from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import peewee
from peewee import IntegrityError

from resp_db import sql
from resp_db.common_types import MaybeSequence
from resp_db.logger import LoggerMixin
from resp_db.time_series_utils import find_peaks, resample_time_series


class RpmDatabaseClient(LoggerMixin):
    """database client to perform querying and preprocessing."""

    def __init__(self, db_filepath: Path = "rpm_signals.db"):
        self.db_filepath = db_filepath
        if not self.db_filepath.is_file():
            raise FileNotFoundError(f"{self.db_filepath} is not a valid database file!")
        sql.database.init(database=self.db_filepath)
        self._create_tables()

    def __repr__(self):
        return f"<{self.__class__.__name__}(datapath={self.db_filepath!r})>"

    def __enter__(self):
        self.logger.debug(f"{self} successfully connected!")
        sql.database.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _create_tables(self):
        with self:
            sql.database.create_tables(
                [
                    sql.Signal,
                    sql.ResearchNumber,
                    sql.Patient,
                    sql.RespiratoryStats,
                    sql.DeepLearningDataset,
                ],
                safe=True,
            )

    @staticmethod
    def _combination_repr(
        research_number: int, modality: str, fraction: int, origin: str
    ) -> str:
        combination_repr = (
            f"\n"
            f"\033[4mENTERED INPUT PARAMETERS\033[0m \n"
            f"ORIGIN : {origin} \n"
            f"RESEARCH NUMBER: {research_number} \n"
            f"MODALITY: {modality} \n"
            f"FRACTION: {fraction}"
            f"\n"
        )
        return combination_repr

    @staticmethod
    def _check_function_inputs(
        research_number: int, modality: str | None, fraction: int | None
    ):
        possible_modalities = ["4DCT", "CBCT", "LINAC"]
        if modality and modality not in possible_modalities:
            raise ValueError(f"{modality=} has to be one of {possible_modalities}.")
        if fraction and not 0 <= fraction < 11:
            raise ValueError(
                f"{fraction=} is an invalid input. It has to be between 0 and 10"
            )

        research_number_str = str(research_number)
        if not any(
            [research_number_str.startswith("57"), len(research_number_str) == 7]
        ):
            raise ValueError(f"{research_number} has to start with 57 and has 7 digits")

    @staticmethod
    def get_specific_signal(
        research_number: int,
        modality: str | None,
        fraction: int | None,
        origin: str = "UKE",
        return_only_query: bool = False,
    ) -> tuple[MaybeSequence[pd.DataFrame], MaybeSequence[sql.Signal]] | MaybeSequence[
        sql.Signal
    ]:

        RpmDatabaseClient._check_function_inputs(research_number, modality, fraction)
        query = (
            sql.Signal.select(sql.Signal)
            .join(
                sql.ResearchNumber,
                on=(sql.Signal.research_number == sql.ResearchNumber.id),
            )
            .join(sql.Patient, on=(sql.ResearchNumber.patient_id == sql.Patient.id))
            .where(
                sql.ResearchNumber.id == research_number,
                sql.Patient.origin == origin,
            )
        )
        if fraction is not None:
            query = query.where(sql.Signal.fraction == fraction)
        if modality:
            query = query.where(sql.Signal.modality == modality)
        if query.count() == 0:
            raise FileNotFoundError(
                f"No signal found in database for the following combination:"
                f"{RpmDatabaseClient._combination_repr(research_number, modality, fraction, origin)}"
            )
        if return_only_query:
            return query
        if query.count() > 1 and fraction is not None and modality is not None:
            raise IntegrityError(
                f"Database fuck-up. {query.count()} signals were found for an unique combination. \n"
                f"Please review and clean database for:"
                f"{RpmDatabaseClient._combination_repr(research_number, modality, fraction, origin)}"
            )
        df_signals = [pickle.loads(signal.df_signal) for signal in query]
        if query.count() == 1:
            return df_signals[0], query[0]
        return df_signals, query

    @staticmethod
    def preprocess_signal(
        df_signal: pd.DataFrame | bytes,
        only_beam_on: bool = True,
        sampling_rate: int = 25,
        remove_offset: bool = True,
    ) -> pd.DataFrame:
        """Performs preprocessing by.

        - only using first to last beam on point (excluding potential acquisition errors)
        - resampling to given sampling_rate
        - shifting raw signal that first three minima are at zero.
        :param df_signal:
        :param only_beam_on:
        :param sampling_rate:
        :param remove_offset:
        :return: pd.Dataframe
        """
        if isinstance(df_signal, bytes):
            df_signal = pickle.loads(df_signal)
        if not isinstance(df_signal, pd.DataFrame):
            raise ValueError(
                f"df_signal should be a Dataframe but is type {type(df_signal)}"
            )
        if not {"time", "amplitude", "beam_on"}.issubset(df_signal.columns):
            raise ValueError(
                f"Dataframe does not contain all mandatory columns; {df_signal.columns}"
            )
        if (
            any(df_signal.amplitude.isna())
            or any(df_signal.time.isna())
            or any(df_signal.beam_on.isna())
        ):
            raise ValueError("Contain invalid data")
        if only_beam_on:
            beam_on_idx = np.where(df_signal.beam_on == 0)[0]
            first_beam_on, last_beam_on = min(beam_on_idx), max(beam_on_idx)
            df_signal = df_signal[first_beam_on:last_beam_on]
            df_signal.reset_index(inplace=True, drop=True)
            time_offset = df_signal.time.min()
            df_signal[:]["time"] -= time_offset
        if sampling_rate:
            t_new, a_new = resample_time_series(
                signal_time_secs=df_signal.time.values,
                signal_amplitude=df_signal.amplitude.values,
                target_samples_per_second=sampling_rate,
            )
            df_signal = pd.DataFrame.from_dict(
                dict(time=t_new, amplitude=a_new), dtype=float
            )
        if remove_offset:
            signal_subset = -1 * df_signal.amplitude[df_signal.time < 50]
            number_minima = 3
            minima_idx = find_peaks(x=signal_subset.values)
            minima = df_signal.amplitude[minima_idx].values
            df_signal.loc[:, "amplitude"] = (
                df_signal.amplitude - minima[:number_minima].mean()
            )
        return df_signal

    @staticmethod
    def get_signals_of_dl_dataset(dl_dataset: str, project: str) -> peewee.ModelSelect:
        if dl_dataset not in ["train", "val", "test"]:
            raise ValueError(f"Invalid dataset: {dl_dataset}")

        query = (
            sql.Signal.select(sql.Signal)
            .join(
                sql.DeepLearningDataset,
                on=(sql.DeepLearningDataset.signal == sql.Signal.id),
            )
            .where(
                sql.DeepLearningDataset.set == dl_dataset,
                sql.DeepLearningDataset.project == project,
                sql.Signal.is_corrupted == 0,
            )
        )
        return query

    def close(self):
        sql.database.close()
        self.logger.debug(f"{self.db_filepath} was disconnected.")
