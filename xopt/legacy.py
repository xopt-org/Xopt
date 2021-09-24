from typing import Dict
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def reformat_config(config: Dict) -> Dict:
    """
    Reformat old config files to enable their use for new versions of xopt.
    Raise a bunch of warnings so it annoys people into updating their config files

    Parameters
    ----------
    config: Dict
        Old config file to be checked

    Returns
    -------
    new_config: Dict
        Updated config file

    """
    # copy config
    new_config = deepcopy(config)

    # check xopt
    if 'algorithm' in new_config['xopt']:
        logger.warning('`algorithm` keyword no longer allowed in xopt config, removing')
        del new_config['xopt']['algorithm']

    if 'verbose' in new_config['xopt']:
        logger.warning('`verbose` keyword no longer allowed in xopt config, renaming '
                       'to `logging`')
        new_config['xopt']['logging'] = logging.INFO if new_config['xopt']['verbose']\
            else logging.WARNING
        del new_config['xopt']['verbose']

    # check simulation
    if 'evaluate' in new_config['simulation']:
        logger.warning('`evaluate` keyword no longer allowed in simulation config, '
                       'moving to `function`')
        new_config['simulation']['function'] = new_config['simulation']['evaluate']
        del new_config['simulation']['evaluate']

    if 'templates' in new_config['simulation']:
        logger.warning('`templates` keyword no longer allowed in simulation config, '
                       'moving to `options`')
        try:
            new_config['simulation']['options'].update({'templates': new_config[
                'simulation']['templates']})
        except KeyError:
            new_config['simulation']['options'] = {'templates': new_config[
                'simulation']['templates']}

        del new_config['simulation']['templates']

    # check vocs
    for ele in ['name', 'description', 'simulation']:
        if ele in new_config['vocs']:
            logger.warning(
                f'`{ele}` keyword no longer allowed in xopt config, removing')
            del new_config['vocs'][ele]

    # move templates to simulation
    if 'templates' in new_config['vocs']:
        logger.warning('`templates` keyword no longer allowed in vocs config, '
                       'moving to simulation `options`')
        try:
            new_config['simulation']['options'].update({'templates': new_config[
                'vocs']['templates']})
        except KeyError:
            new_config['simulation']['options'] = {'templates': new_config[
                'vocs']['templates']}

        del new_config['vocs']['templates']

    return new_config



