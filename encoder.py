import contextlib
import json
import logging
import os
import random
import re
import shutil
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
        bbox=(132, 560, 492, 660),
        max_tokens=500,
        holdout=0.2,
        checkpoint=2000,
        chunksize=100,
        file_prefixes={
            'train': 'train',
            'test': 'test',
            'encodings': 'encodings'
        },
        ext='.nc'
    ):
        self.bbox = bbox
        self.x_dim = bbox[2] - bbox[0]
        self.y_dim = bbox[3] - bbox[1]
        self.datadir = datadir
        self.max_tokens = max_tokens
        self.holdout = holdout
        self.checkpoint = checkpoint
        self.chunksize = chunksize
        self.file_prefixes = file_prefixes
        self.ext = ext
        self.arrays = {}
        self.encodings = {
            'input chars': {'null': 0},
            'tokens': {'null': 0},
            'fonts': {'null': 0}
        }
        self.processed = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def open(self):
        self._load_encoding_dicts()
        self._load_arrays()

    def close(self):
        self._save_encoding_dicts()
        self._save_checkpoint()

    def feed(self, filename):
        self.filename = filename
        root = ET.parse(filename).getroot()
        for snippet in root.iter('snippet'):
            self._parse(snippet)

    def _parse(self, snippet):
        """
        Warning: Currently the tokens matrix has its last entry
        equal to the number of latex tokens parsed.
        """
        try:
            example = self._create_new_example()
            features = example['features']
            tokens = example['tokens']
            for node in snippet.iter('text'):
                self._parse_input_char(features, node)
            self._parse_tokens(tokens, snippet.find('source'))
            if random.random() > self.holdout:
                array_name = 'train'
            else:
                array_name = 'test'
            self.arrays[array_name] = xr.concat(
                [self.arrays[array_name], example], dim='example')
            self.processed = self.processed + 1
            if self.processed > self.checkpoint:
                self._save_checkpoint()
                self._load_arrays()
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
        return self.encodings['input chars'].setdefault(
            char, len(self.encodings['input chars']))

    def _parse_tokens(self, encoded, node):
        tokens = latex.parse_into_tokens(node.text)
        if len(tokens) > self.max_tokens:
            raise InvalidSnippetError("Number of latex tokens exceeds limit")
        for token in tokens:
            code = self._get_token_encoding(token)
            prev_token_pos = encoded[0, -1]
            encoded[0, prev_token_pos+1] = code
            encoded[0, -1] = prev_token_pos + 1

    def _get_token_encoding(self, token):
        return self.encodings['tokens'].setdefault(
            token, len(self.encodings['tokens']))

    def _load_arrays(self):
        for index, fname in self._zip_array_ids_and_filenames():
            try:
                self.arrays[index] = xr.open_mfdataset(
                    fname, chunks={'example': self.chunksize}
                )
            except IOError:
                self.arrays[index] = self._create_new_dataset()

    def _save_checkpoint(self):
        self._save_arrays('tmp-')
        for index in ['train', 'test']:
            self.arrays[index].close()
        for orig, dest in self._zip_checkpoint_and_master_filenames():
            shutil.move(orig, dest)

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
            np.zeros([0, self.max_tokens+1], dtype='int32'),
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
            np.zeros([1, self.max_tokens+1], dtype='int32'),
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

    def _zip_array_ids_and_filenames(self, pre=''):
        arrays = ['train', 'test']
        filenames = [
            os.path.join(
                self.datadir, pre + self.file_prefixes[ftype] + self.ext)
            for ftype in arrays
        ]
        return zip(arrays, filenames)

    def _zip_checkpoint_and_master_filenames(self):
        arrays = ['train', 'test']
        prefixes = ['tmp-', '']
        filenames = [
            [os.path.join(
                self.datadir, pre + self.file_prefixes[ftype] + self.ext)
                for pre in prefixes]
            for ftype in arrays
        ]
        return filenames

    def _encoding_filename(self):
        prefix = self.file_prefixes['encodings']
        return os.path.join(self.datadir, prefix + '.json')


class InvalidSnippetError(Exception):
    """Exception raised when attempting to encode an invalid pdf snippet."""
