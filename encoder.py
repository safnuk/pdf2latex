import contextlib
import glob
import json
import logging
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import xarray as xr

import latex

# FEATURES = ['char', 'font', 'fontsize', 'charsize']
FEATURES = ['char', 'font', 'fontsize']
FEATURE_DIM = len(FEATURES)


class Encoder:
    def __init__(
        self,
        datadir='data/',
        bbox=(132, 560, 498, 660),
        max_tokens=500,
        checkpoint=5000,
        file_prefixes={
            'examples': 'examples',
            'encodings': 'encodings'
        },
        ext='.nc'
    ):
        self.bbox = bbox
        self.x_dim = bbox[2] - bbox[0]
        self.y_dim = bbox[3] - bbox[1]
        self.datadir = datadir
        self.max_tokens = max_tokens
        self.checkpoint = checkpoint
        self.file_prefixes = file_prefixes
        self.ext = ext
        self.examples = None
        self.encodings = {
            'chars': {'NULL': 0},
            'tokens': {'NULL': 0, 'GO': 1, 'STOP': 2},
            'fonts': {'NULL': 0}
        }
        self.processed = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def open(self):
        self._load_encoding_dicts()
        self.examples = self._create_new_dataset()

    def close(self):
        self._save_encoding_dicts()
        self._save_checkpoint()

    def feed(self, filename):
        self.filename = filename
        root = ET.parse(filename).getroot()
        for snippet in root.iter('snippet'):
            self._parse(snippet)

    def _parse(self, snippet):
        try:
            example = self._create_new_example()
            features = example['features']
            tokens = example['tokens']
            for node in snippet.iter('text'):
                self._parse_input_char(features, node)
            self._parse_tokens(tokens, snippet.find('source'))
            self.examples = xr.concat(
                [self.examples, example], dim='example')
            self.processed += 1
            if self.processed >= self.checkpoint:
                self.close()
                self.open()
                self.processed = 0
        except InvalidSnippetError as e:
            logging.warning('Skipped snippet in %s - %s', self.filename, e)

    def _parse_input_char(self, features, node):
        text = re.sub(r'\s+', '', node.text)
        if text == '':
            return
        x, y = self._get_transformed_char_position(node)
        self._confirm_in_bbox(x, y)
        fontcode, fontsize, charsize = self._get_font_encoding_and_sizes(node)
        charcode = self._get_char_encoding(text)
        features.loc[dict(x=x, y=y, feature='char')] = charcode
        features.loc[dict(x=x, y=y, feature='font')] = fontcode
        features.loc[dict(x=x, y=y, feature='fontsize')] = fontsize
        if 'charsize' in FEATURES:
            features.loc[dict(x=x, y=y, feature='charsize')] = charsize

    def _get_transformed_char_position(self, node):
        bbox = node.attrib['bbox']
        coords = [x for x in map(lambda z: int(float(z)), bbox.split(','))]
        x = coords[0] - self.bbox[0]
        y = self.bbox[3] - coords[1]
        return x, y

    def _confirm_in_bbox(self, x, y):
        if not (x >= 0 and y >= 0 and x < self.x_dim and y < self.y_dim):
            raise InvalidSnippetError("Character outside of bounding box")

    def _get_font_encoding_and_sizes(self, node):
        try:
            font_string = node.attrib['font']
            size = round(float(node.attrib['size']))
            font_and_size = font_string.split('+')[1]
            font = re.sub(r'[0-9]+', '', font_and_size)
            fontsize = int(re.sub(r'[A-Z]+', '', font_and_size))
            fontcode = self._get_font_encoding(font)
        except (IndexError, ValueError):
            raise InvalidSnippetError("Could not parse font")
        return fontcode, fontsize, size

    def _get_font_encoding(self, font):
        return self.encodings['fonts'].setdefault(
            font, len(self.encodings['fonts']))

    def _get_char_encoding(self, char):
        return self.encodings['chars'].setdefault(
            char, len(self.encodings['chars']))

    def _parse_tokens(self, encoded, node):
        tokens = latex.parse_into_tokens(node.text)
        if len(tokens) > self.max_tokens:
            raise InvalidSnippetError("Number of latex tokens exceeds limit")
        encoded[0, 0] = self._get_token_encoding('GO')
        encoded[0, -1] = 1
        for token in tokens:
            code = self._get_token_encoding(token)
            num_tokens_parsed = encoded[0, -1]
            encoded[0, num_tokens_parsed] = code
            encoded[0, -1] += 1
        end = encoded[0, -1]
        encoded[0, end] = self._get_token_encoding('STOP')
        encoded[0, -1] += 1

    def _get_token_encoding(self, token):
        return self.encodings['tokens'].setdefault(
            token, len(self.encodings['tokens']))

    def _save_checkpoint(self):
        self.examples.to_netcdf(
            self._get_next_examples_filename(),
            format='netCDF4', engine='netcdf4',
            encoding={
                'features': {'zlib': True, 'complevel': 9},
                'tokens': {'zlib': True, 'complevel': 9}
            },
        )

    def _save_arrays(self, pre=''):
        for index, fname in self._zip_array_ids_and_filenames(pre):
            self.arrays[index].close()
            self.arrays[index].to_netcdf(
                fname, format='netCDF4', engine='netcdf4',
                encoding={
                    'features': {'zlib': True, 'complevel': 9},
                    'tokens': {'zlib': True, 'complevel': 9}
                },
                unlimited_dims=['example'],
            )

    def _create_new_dataset(self):
        features = xr.DataArray(
            np.zeros([0, self.x_dim, self.y_dim, FEATURE_DIM], dtype='int32'),
            coords={'feature': FEATURES},
            dims=('example', 'x', 'y', 'feature'),
        )
        tokens = xr.DataArray(
            np.zeros([0, self.max_tokens+3], dtype='int32'),
            dims=('example', 'token'),
        )
        return xr.Dataset({'features': features, 'tokens': tokens})

    def _create_new_example(self):
        features = xr.DataArray(
            np.zeros([1, self.x_dim, self.y_dim, FEATURE_DIM], dtype='int32'),
            coords={'feature': FEATURES},
            dims=('example', 'x', 'y', 'feature'),
        )
        tokens = xr.DataArray(
            np.zeros([1, self.max_tokens+3], dtype='int32'),
            dims=('example', 'token'),
        )
        return xr.Dataset({'features': features, 'tokens': tokens})

    def _load_encoding_dicts(self):
        with contextlib.suppress(FileNotFoundError):
            with open(self._encoding_filename(), 'r') as f:
                self.encodings = json.load(f)

    def _save_encoding_dicts(self):
        with open(self._encoding_filename(), 'w') as f:
            json.dump(self.encodings, f)

    def _encoding_filename(self):
        prefix = self.file_prefixes['encodings']
        return os.path.join(self.datadir, prefix + '.json')

    def _get_next_examples_filename(self):
        file_format = self.file_prefixes['examples'] + '{:03d}' + self.ext
        match = os.path.join(self.datadir,
                             self.file_prefixes['examples'] + '*' + self.ext)
        number_of_files = len(glob.glob(match))
        return os.path.join(self.datadir, file_format.format(number_of_files))


class InvalidSnippetError(Exception):
    """Exception raised when attempting to encode an invalid pdf snippet."""
