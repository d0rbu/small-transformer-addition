import logging
import numpy as np

def log_config(arguments):
    logging.info("Logging used config:")
    logging.info("-" * 50)
    for argument, value in arguments.items():
        logging.info("{}: {}".format(argument, value))
    logging.info("-" * 50)


def compute_pass_at_k(num_samples, num_correct, k):
    if num_samples - num_correct < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))