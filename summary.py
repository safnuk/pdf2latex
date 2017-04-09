import math
import re
import xml.etree.ElementTree as ET

import pandas as pd

from arxiv_parser import get_filenames
from latex import parse_into_tokens

BASE = '/data/safnu1b/latex/'
# BASE = ''
DATADIR = BASE + 'data/snippets'


class DataStats:
    def __init__(self, datadir, bbox_ll=(132, 560), bbox_ur=(492, 660)):
        self.bbox_ll = bbox_ll
        self.bbox_ur = bbox_ur
        self.in_box = 0
        self.out_box = 0
        self.datadir = datadir
        self.fonts = set()
        self.fontsizes = set()
        self.fontsizes_type = set()
        self.input_chars = set()
        self.bbox_min = [math.inf, math.inf]
        self.bbox_max = [-math.inf, -math.inf]
        self.latex_tokens = set()
        self.snip_lengths = []

    def calc_stats(self):
        files = get_filenames(self.datadir, '.xml')
        for filename in files:
            root = ET.parse(filename).getroot()
            for snip in root.iter('snippet'):
                input_count = 0
                token_count = 0
                out_of_bounds = False
                for node in snip.iter('text'):
                    if node.text:
                        input_count = input_count + 1
                    self.update_input_chars(node)
                    if not self.update_bbox(node):
                        out_of_bounds = True
                    self.update_fonts(node)
                if out_of_bounds:
                    self.out_box = self.out_box + 1
                else:
                    self.in_box = self.in_box + 1
                source = snip.find('source')
                token_count = self.update_tokens(source)
                self.snip_lengths.append([input_count, token_count])

    def update_tokens(self, node):
        if node.text:
            tokens = parse_into_tokens(node.text)
            self.latex_tokens = self.latex_tokens.union(tokens)
            return len(tokens)
        return 0

    def update_fonts(self, node):
        try:
            if 'font' in node.attrib:
                font = node.attrib['font']
                font_and_size = font.split('+')[1]
                font = re.sub(r'[0-9]+', '', font_and_size)
                size = re.sub(r'[A-Z]+', '', font_and_size)
                self.fontsizes_type.add(float(size))
                self.fonts.add(font)
            if 'size' in node.attrib:
                self.fontsizes.add(node.attrib['size'])
        except (IndexError, ValueError):
            print("Trouble parsing {}".format(node.attrib['font']))

    def update_bbox(self, node):
        if 'bbox' not in node.attrib:
            return True
        bbox = node.attrib['bbox']
        coords = [x for x in map(float, bbox.split(','))]
        if coords[0] < self.bbox_min[0]:
            self.bbox_min[0] = coords[0]
        if coords[1] < self.bbox_min[1]:
            self.bbox_min[1] = coords[1]
        if coords[0] > self.bbox_max[0]:
            self.bbox_max[0] = coords[2]
        if coords[1] > self.bbox_max[1]:
            self.bbox_max[1] = coords[3]

        if (coords[0] < self.bbox_ll[0] or
                coords[1] < self.bbox_ll[1] or
                coords[0] > self.bbox_ur[0] or
                coords[1] > self.bbox_ur[1]):
            return False
        return True

    def update_input_chars(self, node):
        if node.text:
            self.input_chars.add(node.text)

    def print_stats(self):
        print("Bounding box: {} {}".format(self.bbox_min, self.bbox_max))
        print("Number of fonts: {}".format(len(self.fonts)))
        print("Number of font sizes: {} Range: ({}, {})".format(
            len(self.fontsizes), min(map(float, self.fontsizes)),
            max(map(float, self.fontsizes))
        ))
        print("Number of font type sizes: {} Range: ({}, {})".format(
            len(self.fontsizes_type), min(map(float, self.fontsizes_type)),
            max(map(float, self.fontsizes_type))
        ))
        print('Number of input chars: {}'.format(len(self.input_chars)))
        print('Number of latex tokens: {}'.format(len(self.latex_tokens)))
        print('Snippets in bbox: {}; Snippets outside bbox: {}'.format(
            self.in_box, self.out_box
        ))
        data = pd.DataFrame(self.snip_lengths)
        m = data.max()
        print("Max input stream length: {} Max token stream length: {}".format(
            m[0], m[1]
        ))
        # print('-----------------------------')
        # input_chars = [x for x in self.input_chars]
        # input_chars.sort()
        # tokens = [x for x in self.latex_tokens]
        # tokens.sort()
        # print('Input chars: {}'.format(' '.join(input_chars)))
        # print('-----------------------------')
        # print('Latex tokens: {}'.format(' '.join(tokens)))

if __name__ == '__main__':
    summary = DataStats(DATADIR)
    summary.calc_stats()
    summary.print_stats()
