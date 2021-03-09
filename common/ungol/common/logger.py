# -*- coding: utf-8 -*-

import os
import pathlib
import logging
from logging.config import fileConfig


def get(name: str) -> logging.Logger:
    return logging.getLogger(f'ungol.{name}')


# be nice if used as a library - do not log to stderr as default
logging.getLogger('ungol').addHandler(logging.NullHandler())

# probe environment for logging configuration:
#   1. if conf/logging.conf exists use this
#   2. if UNGOL_LOG is set as environment variable use its value

f_conf = None

path = pathlib.Path('conf/logging.conf')
if path.is_file():
    f_conf = str(path)

if 'UNGOL_LOG' in os.environ:
    f_conf = str(os.environ['UNGOL_LOG'])

if f_conf is not None:
    fileConfig(f_conf)


log = get('common.logger')
log.info('initialized logging')


# disabling others
for mod in ('summarizer', ):
    log.info(f'disabling logging for {mod}.*')
    logging.getLogger(mod).setLevel(logging.WARNING)
