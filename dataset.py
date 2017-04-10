import json
import os

import xarray as xr


class Dataset:
    def __init__(self, data):
        self.data = data
        self.num_examples = len(data['example'])

        self.epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self.data.features

    @property
    def tokens(self):
        return self.data.tokens

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if start + batch_size >= self.num_examples:
            self.epochs_completed += 1
            start = 0
            self._index_in_epoch = 0
            # TODO: Include the examples skipped over at the tail end
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        batch = self.data[dict(example=slice(start, end))]
        return batch.features.values, batch.tokens.values


class Datasets:
    def __init__(self, train, validate, test, encodings):
        self.train = train
        self.validate = validate
        self.test = test
        self.encodings = encodings


def read_datasets(data_dir, validation_size=10000):
    TRAIN_PDFS = 'train.nc'
    TEST_PDFS = 'test.nc'
    ENCODINGS = 'encodings.json'

    train_pdfs = _load_examples_file(os.path.join(data_dir, TRAIN_PDFS))
    test_pdfs = _load_examples_file(os.path.join(data_dir, TEST_PDFS))
    with open(os.path.join(data_dir, ENCODINGS), 'r') as f:
        encodings = json.load(f)
    if not 0 <= validation_size <= len(train_pdfs['example']):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_pdfs['example']), validation_size))
    train = Dataset(train_pdfs[dict(example=slice(None, validation_size))])
    validation = Dataset(train_pdfs[dict(example=slice(validation_size, None))])
    test = Dataset(test_pdfs)
    return Datasets(train, validation, test, encodings)


def _load_examples_file(filename, chunksize=100):
    return xr.open_mfdataset(filename, chunks={'example': chunksize})
