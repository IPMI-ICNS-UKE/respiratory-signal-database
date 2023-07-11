import logging
from pathlib import Path

from resp_db.client import RpmDatabaseClient
from resp_db.logger import init_fancy_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    init_fancy_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger("resp_db").setLevel(logging.DEBUG)
    database_root = Path(".../open_access_rpm_signals_master.db")  # change to path of downloaded database
    client = RpmDatabaseClient(db_filepath=database_root)

    # get one specific signal and preprocess
    with client:
        df_signal, found_signal = client.get_specific_signal(
            research_number=5708019, fraction=0, modality="4DCT"
        )
    df_signal = RpmDatabaseClient.preprocess_signal(
        df_signal=df_signal,
        sampling_rate=25,
        only_beam_on=True,
        remove_offset=True,
    )

    # get all signals of train(/val/test) set.
    with client:
        query = client.get_signals_of_dl_dataset(
            dl_dataset="train", # "val", "test"
            project="short-term-prediction"
        )
    query = list(query)
    # preprocess first signal of train set
    df_signal = RpmDatabaseClient.preprocess_signal(
        df_signal=query[0].df_signal,
        sampling_rate=25,
        only_beam_on=True,
        remove_offset=True,
    )
