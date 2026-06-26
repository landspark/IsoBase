#! python3
# -*- encoding: utf-8 -*-
"""
@File   : test_logger.py
@Created: 2025/04/01 16:17
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""
from isobase import LOGGER

# Here put the import lib.

def test_logger():
    LOGGER.debug("Test IsoBase logger debug.")
    LOGGER.info("Test IsoBase logger info.")
    LOGGER.warning("Test IsoBase logger warning.")
    LOGGER.error("Test IsoBase logger error.")
    LOGGER.exception("Test IsoBase logger exception.")
    LOGGER.critical("Test IsoBase logger critical.")
