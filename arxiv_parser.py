import logging
import os
from shutil import copyfile
import tarfile

from latex import LatexParser

BASE = '/data/safnu1b/latex/'
# BASE = ''
# POST = '2'
POST = ''
IN = BASE + 'data/raw'
OUT = BASE + 'data/processed' + POST
SNIPPED = BASE + 'data/snippets'
COMPILE_DIR = '/tmp/run' + POST
TEMPLATE = 'data/template.tex'
LOG_FILE = BASE + 'data/error.log'
ERASE_WHEN_FINISHED = True


def create_snippets(indir, outdir, logger):
    files = get_filenames(indir)
    for filename in files:
        logger.info("Starting to process %s", filename)
        try:
            _, name = os.path.split(filename)
            name, _ = os.path.splitext(name)
            name = name + '.xml'
            outname = os.path.join(outdir, name)
            latex = LatexParser(filename)
            if latex.is_valid():
                latex.clean()
                latex.compile_snippets(
                    snipfile=outname,
                    preamblefile=TEMPLATE,
                    texdir=COMPILE_DIR
                )
        except Exception:  # skip any files that cause exceptions
            logger.error("Could not process %s", filename)
        finally:
            if ERASE_WHEN_FINISHED:
                os.remove(filename)


def transfer_good_texfiles(indir, outdir):
    files = get_filenames(indir)
    for filename in files:
        tar = try_to_read_tarfile(filename)
        if tar and single_tex_file(tar):
            extract_first_tex_file(tar, outdir)
        else:
            newfilename = os.path.join(outdir,
                                       os.path.basename(filename)) + ".tex"
            copyfile(filename, newfilename)
        if ERASE_WHEN_FINISHED:
            os.remove(filename)


def get_filenames(dir, ext=''):
    return [os.path.join(dir, f) for f in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, f)) and f.endswith(ext)]


def try_to_read_tarfile(filename):
    try:
        return tarfile.open(filename)
    except tarfile.ReadError:
        return None


def single_tex_file(tar):
    count = 0
    for name in tar.getnames():
        _, ext = os.path.splitext(name)
        if ext == '.tex':
            count = count + 1
    return count == 1


def extract_first_tex_file(tar, outdir):
    for f in tar.getmembers():
        if f.isfile():
            _, ext = os.path.splitext(f.name)
            if ext == '.tex':
                tar.extract(f, outdir)


def configure_logger():
    logger = logging.getLogger('runlog')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

if __name__ == '__main__':
    logger = configure_logger()
    logger.info('Started parsing files')
    transfer_good_texfiles(IN, OUT)
    create_snippets(OUT, SNIPPED, logger)
    logger.info('Finished parsing files')
