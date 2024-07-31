import logging

from classes import main_runparams_cls


def set_loggers(
        main_runparams: main_runparams_cls.MainRunParams
) -> None:

    # Setting up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(main_runparams.log_dir / 'SWIS_task.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return None

if __name__ == '__main__':

    main_runparams = main_runparams_cls.MainRunParams()


