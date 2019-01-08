import logging

def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger
    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(levelname)s::%(message)s")
    # formatter = logging.Formatter("%(asctime)s:%(filename)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger

logger = get_logger()


def read_log_inps_test(log_file):
    with open(log_file, 'r') as f:
        inps_tests = []
        for line in f.readlines():
            if 'inps_test' in line:
                string = line.split(':')[-1].strip()
                string = string[1:-1]
                inps_test = float(string.split(',')[0])
                inps_tests.append(inps_test)
    print(inps_tests)
    return inps_tests

def read_log_inps_baseline(log_file):
    with open(log_file, 'r') as f:
        curve = []
        for line in f.readlines():
            if 'inps_baseline:' in line and not '(' in line:
                string = line.split(':')[-1].strip()
                curve.append(float(string))
    return curve

if __name__ == '__main__':
    read_log_inps_baseline('log_5-14/dcgan_cifar10_exp01_baseline.log')


