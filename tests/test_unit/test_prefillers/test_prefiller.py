import logging
import os
import unittest
from unittest import mock

import pandas as pd
import pytest

from src.prefillers.prefilling_additional import shuffle_and_reverse, PreFillerAdditional
from tests.test_integration.test_data import SAMPLE_LIST_FROM_DB

method_options = ["tfidf", "word2vec", "doc2vec", "lda"]
full_text_options = [True, False]
random_reverse_options = [True, False]

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# NOTICE: Logging didn't work really well for Pika so far... That's way using prints.
log_format = '[%(asctime)s] [%(levelname)s] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logging.debug("Testing logging from %s." % os.path.basename(__file__))


def test_shuffle_and_reverse():
    assert isinstance(shuffle_and_reverse(SAMPLE_LIST_FROM_DB, random_order=False), list)


def mock_start_preprocessed_features_prefilling(self, skip_already_filled, random_order):
    raise SystemExit(1)


class PrefillingAdditional(unittest.TestCase):

    def test_fill_body_preprocessed(self):
        with mock.patch.object(PreFillerAdditional, 'fill_body_preprocessed',
                               new=mock_start_preprocessed_features_prefilling):
            with pytest.raises(SystemExit):
                PreFillerAdditional().fill_body_preprocessed(True, False)

