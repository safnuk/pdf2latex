import shutil

import arxiv_parser as ap
import encoder

# BASE = '/data/safnu1b/latex/'
BASE = ''
INDIR = BASE + 'data/snippets'
MOVETO_DIR = BASE + 'data/done'
OUTDIR = BASE + 'data'

MOVE_FILES = True


def move_file(filename, dest_dir):
    shutil.move(filename, dest_dir)

if __name__ == "__main__":
    filenames = ap.get_filenames(INDIR, '.xml')
    with encoder.Encoder(OUTDIR) as e:
        for f in filenames:
            e.feed(f)
            if MOVE_FILES:
                move_file(f, MOVETO_DIR)
