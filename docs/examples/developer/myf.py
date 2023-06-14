

import logging
logger = logging.getLogger(__name__)


def f():
    print('__name__ =', __name__)
    logger.info('this is f')
