import logging
import os
from datetime import datetime


def logger_init(file_log=False, file_name=''):

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(module)s line:%(lineno)d %(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    if file_log:
        formatter = logging.Formatter('%(levelname)s %(module)s line:%(lineno)d %(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
        log_file_dir = './log_file'
        if not os.path.exists(log_file_dir): os.makedirs(log_file_dir)
        filehdlr = logging.FileHandler('./log_file/log-'+datetime.now().strftime('%Y-%m-%d %I-%M-%S %p ')+file_name)
        filehdlr.setFormatter(formatter)
        filehdlr.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(filehdlr)

def get_logger(name='logger_name', console_log=True, file_log=False, filename=''):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = '%(asctime)s (%(levelname)s) %(module)s line:%(lineno)d %(message)s'
    formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')

    if console_log:
        console = logging.StreamHandler()
        # console.setLevel(logging.FATAL)
        console.setFormatter(formatter)
        logger.addHandler(console)

    if file_log:
        if not os.path.exists('./log_file/'): os.makedirs('./log_file/')
        filehdlr = logging.FileHandler('./log_file/log-'+datetime.now().strftime('%Y-%m-%d %H-%M-%S ')+filename)
        # filehdlr.setLevel(logging.DEBUG)
        filehdlr.setFormatter(formatter)
        logger.addHandler(filehdlr)

    return logger