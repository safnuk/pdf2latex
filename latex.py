import contextlib
from enum import Enum
import logging
import os
import string
import subprocess
import xml.etree.ElementTree as ET

from chardet.universaldetector import UniversalDetector

DEFAULT_SUBSTITUTIONS = {
    '\\begin{displaymath}': '\\begin{equation*}',
    '\\end{displaymath}': '\\end{equation*}',
    '\\[': '\\begin{equation*}',
    '\\]': '\\end{equation*}',
    '\\(': '$', '\\)': '$',
    '\\begin{math}': '$', '\\end{math}': '$',
}

BANNED_KEYWORDS = [
    '\\label', '\\ref', '\\eqref', '\\cite', '\\pagestyle',
    '\\thispagestyle', '\\title', '\\author', '\\address',
    '\\email', '\\subjclass', '\\thanks', '\\keywords',
    '\\tableofcontents', '\\includegraphics',
    '\\listoffigures', '\\listoftables',
    '\\bibliographystyle', '\\bibliography', '\\maketitle',
    '\\numberwithin', '\\theoremstyle', '\\allowdisplaybreaks',
    '\\setcounter', '\\newpage', '\\pagebreak', '\\clearpage',
    '\\input', '\\include', '\\includeonly', '\\footnote',
    '\\footnotesize', '\\citep'
]

BANNED_ENVIRONMENTS = [
    'figure', 'tikzpicture', 'keywords', 'AMS', 'table', 'tikzcd',
    'figure*',
]

MIN_LENGTH = 100
MAX_LENGTH = 500


class LatexParser:

    def __init__(self, filename):
        self.filename = filename
        # with open(filename, encoding=self.detect_encoding()) as f:
        with open(filename, encoding='latin-1') as f:
            self.lines = [line.strip() for line in f]
        self.macros = dict(DEFAULT_SUBSTITUTIONS)
        self.macro_args = {}
        for cmd in self.macros:
            self.macro_args[cmd] = (0, False, '')
        self.theorems = []
        self.simplify()

    def simplify(self):
        self.remove_comments()
        self.remove_unneeded_linebreaks()

    def clean(self):
        self.expand_macros()
        self.remove_banned_keywords()
        self.replace_double_dollars()
        self.remove_preamble()
        self.remove_unneeded_linebreaks()
        self.structure_environments()
        self.remove_banned_environments()
        self.remove_duplicate_linebreaks()
        self.remove_empty_environments()
        self.join_each_environment()
        self.make_lines_fit_length()

    def is_valid(self):
        head = self.lines[0]
        return ('\\documentclass' in head) or ('\\documentstyle' in head)

    def save(self, filename=None):
        filename = filename if filename else self.filename
        with open(filename, 'w') as f:
            for line in self.lines:
                print(line, file=f)

    def compile_snippets(
        self, snipfile='data/good.xml', preamblefile='data/template.tex',
        texdir='data/run', timeout=2
    ):
        logger = logging.getLogger('runlog')
        savefile = os.path.join(texdir, 'test.tex')
        auxfile = os.path.join(texdir, 'test.aux')
        logfile = os.path.join(texdir, 'test.log')
        with open(preamblefile) as f:
            text = ''.join(f)
        good = ET.Element('file')
        good.set('name', self.filename)
        good_count = 0
        bad_count = 0
        for line in self.lines:
            with open(savefile, 'w') as f:
                print(text % line, file=f)
            with contextlib.suppress(FileNotFoundError):
                os.remove(auxfile)
                os.remove(logfile)
            code = attempt_compile(savefile, timeout)
            if code == CompileCodes.OK:
                try:
                    node = create_xml_from_pdf(texdir)
                    if len(node) > 1:
                        bad_count = bad_count + 1
                        logger.warning("PDF larger than 1 page in %s",
                                       self.filename)
                    else:
                        snippet = ET.SubElement(good, 'snippet')
                        pdf = ET.SubElement(snippet, 'pdf')
                        for textnode in node.iter('text'):
                            pdf.append(textnode)
                        source = ET.SubElement(snippet, 'source')
                        source.text = line
                        good_count += 1
                except Exception:
                    bad_count += 1
                    logger.error("Error with xml parsing in %s", self.filename)
                    continue
            else:
                bad_count = bad_count + 1
        tree = ET.ElementTree(good)
        tree.write(snipfile)
        logger.info("File: %s Good snippets: %d Rejected: %d", self.filename,
                    good_count, bad_count)

    def remove_preamble(self):
        begin = '\\begin{document}'
        end = '\\end{document}'
        in_preamble = True
        new_lines = []
        for line in self.lines:
            if in_preamble:
                if self.contains_keyword(line, begin):
                    in_preamble = False
                    start = self.keyword_index(line, begin)
                    new_lines.append(begin)
                    if line[start + len(begin):]:
                        new_lines.append(line[start + len(begin):])
            elif self.contains_keyword(line, end):
                finish = self.keyword_index(line, end)
                if line[:finish]:
                    new_lines.append(line[:finish])
                new_lines.append(end)
                break
            else:
                new_lines.append(line)
        self.lines = new_lines

    def make_lines_fit_length(
        self, min_length=MIN_LENGTH, max_length=MAX_LENGTH
    ):
        new_lines = []
        cur_join = ''
        for line in self.lines:
            if not cur_join:
                cur_join = line
            else:
                cur_join = cur_join + '\n' + line
            if len(cur_join) > min_length and len(cur_join) < max_length:
                new_lines.append(cur_join)
                cur_join = ''
            if len(cur_join) >= max_length:  # discard too long lines
                cur_join = ''
        if cur_join:
            new_lines.append(cur_join)
        self.lines = new_lines

    def structure_environments(self):
        keywords = ['\\begin', '\\end']
        for keyword in keywords:
            new_lines = []
            for line in self.lines:
                    while self.contains_keyword(line, keyword):
                        start = self.keyword_index(line, keyword)
                        if line[:start]:
                            new_lines.append(line[:start].strip())
                        line = line[start:]
                        start, finish = self.find_matching_brackets(
                            line, '{', '}')
                        new_lines.append(keyword + line[start:finish+1])
                        line = line[finish+1:].strip()
                    if line:
                        new_lines.append(line)
            self.lines = new_lines
        # remove \begin{document}, \end{document}
        self.lines = self.lines[1:-1]

    def join_each_environment(self):
        new_lines = []
        depth = 0
        for line in self.lines:
            if depth == 0:
                cur_join = line
            else:
                cur_join = cur_join + '\n' + line
            if self.contains_keyword(line, '\\begin'):
                depth = depth + 1
            elif self.contains_keyword(line, '\\end'):
                depth = depth - 1
            if depth == 0 and cur_join:
                new_lines.append(cur_join)
        self.lines = new_lines

    def remove_empty_environments(self):
        n = 1
        while n < len(self.lines):
            if (self.contains_keyword(self.lines[n], '\\end') and
                    self.contains_keyword(self.lines[n-1], '\\begin')):
                del self.lines[n-1:n+1]
                n = n - 1
            else:
                n = n + 1

    def remove_banned_environments(self):
        new_lines = []
        banned_depth = 0
        for line in self.lines:
            if self.is_banned_env(line, '\\begin'):
                banned_depth = banned_depth + 1
            elif self.is_banned_env(line, '\\end'):
                banned_depth = banned_depth - 1
            elif banned_depth == 0:
                new_lines.append(line)
        self.lines = new_lines

    def is_banned_env(self, line, cmd):
        if not line.startswith(cmd):
            return False
        for key in BANNED_ENVIRONMENTS:
            text = cmd + '{%s}' % key
            if line.startswith(text):
                return True
        return False

    def replace_double_dollars(self):
        # TODO: we should deal with \$$
        begin = '\\begin{equation*}'
        end = '\\end{equation*}'
        current, other = begin, end
        new_lines = []
        for line in self.lines:
            while '$$' in line:
                start = line.find('$$')
                line = line[:start] + current + line[start+2:]
                current, other = other, current
            new_lines.append(line)
        self.lines = new_lines

    def detect_encoding(self):
        detector = UniversalDetector()
        with open(self.filename, 'rb') as f:
            for line in f:
                detector.feed(line)
                if detector.done:
                    break
            detector.close()
            return detector.result['encoding']

    def remove_comments(self):
        commentfree = []
        for line in self.lines:
            if len(line) > 0 and line[0] == '%':
                continue
            found_comment = False
            for n, char in enumerate(line):
                if char == '%' and line[n-1] != '\\':
                    commentfree.append(line[:n])
                    found_comment = True
                    break
            if not found_comment:
                commentfree.append(line)
        self.lines = commentfree

    def remove_unneeded_linebreaks(self):
        while self.lines[0] == '':
            self.lines.pop(0)
        while self.lines[-1] == '':
            self.lines.pop(-1)
        joined_lines = []
        current_join = ''
        for line in self.lines:
            line = line.strip()
            if line == '':
                if current_join:
                    joined_lines.append(current_join)
                    joined_lines.append('')
                    current_join = ''
            else:
                current_join = (current_join + ' ' + line).strip()
        if current_join:
            joined_lines.append(current_join)
        self.lines = joined_lines

    def remove_duplicate_linebreaks(self):
        while self.lines[0] == '':
            self.lines.pop(0)
        while self.lines[-1] == '':
            self.lines.pop(-1)
        new_lines = []
        prev_line = ''
        for line in self.lines:
            if line or prev_line:
                new_lines.append(line)
            prev_line = line
        self.lines = new_lines

    def expand_macros(self):
        new_lines = []
        for line in self.lines:
            line = self.pull_out_newcommands(line)
            line = self.pull_out_newtheorems(line)
            self.fully_expand_replacements()
            new_lines.append(self.expand_macros_in_line(line))
        self.lines = new_lines

    def remove_banned_keywords(self):
        new_lines = []
        for line in self.lines:
            for keyword in BANNED_KEYWORDS:
                while self.contains_keyword(line, keyword):
                    cmd_start = self.keyword_index(line, keyword)
                    cmd_end = cmd_start + len(keyword)
                    head = line[:cmd_start]
                    tail = line[cmd_end:]
                    while tail.startswith('{') or tail.startswith('['):
                        if tail.startswith('{'):
                            tail, _ = self.cut_out_delimited(tail, '{', '}')
                        else:
                            tail, _ = self.cut_out_delimited(tail, '[', ']')
                    line = head + tail
            new_lines.append(line)
        self.lines = new_lines

    def fully_expand_replacements(self):
        MAX_DEPTH = 20
        new_dict = {}
        self.remove_redundant_replacements()
        for target, expr in self.macros.items():
            # print("Replacing {} with {}".format(target, expr))
            depth = 0
            modified = True
            while modified and depth < MAX_DEPTH:
                depth = depth + 1
                modified = False
                for cmd in self.macros:
                    while (self.contains_keyword(expr, cmd) and
                           depth < MAX_DEPTH):
                        depth = depth + 1
                        modified = True
                        expr = self.expand_once(expr, cmd)
            assert depth < MAX_DEPTH, "Max macro expansion depth exceeded"
            if target != expr:
                new_dict[target] = expr
        self.macros = new_dict

    def remove_redundant_replacements(self):
        new_dict = {}
        for cmd, expr in self.macros.items():
            if cmd != expr:
                new_dict[cmd] = expr
        self.macros = new_dict

    def expand_once(self, expr, target):
        replacement = self.macros[target]
        num_of_args, optional, opt_arg = self.macro_args[target]
        args_grabbed = 0
        args = [''] * num_of_args
        if optional:
            args[0] = opt_arg
            args_grabbed = 1
        start = self.keyword_index(expr, target)
        alpha_num = string.ascii_letters + string.digits
        if start == -1:
            return expr
        end = start + len(target)
        prev = expr[start-1:start]
        if prev in alpha_num and replacement[0:1] in alpha_num:
            replacement = ' ' + replacement
        head = expr[:start]
        tail = expr[end:]
        if num_of_args > 0:
            if tail.startswith('['):
                tail, arg = self.cut_out_delimited(tail, '[', ']')
                args[0] = arg
            while args_grabbed < num_of_args:
                tail, arg = self.cut_out_delimited(tail, '{', '}')
                args[args_grabbed] = arg
                args_grabbed = args_grabbed + 1
            for n in range(num_of_args):
                key = '#{}'.format(n+1)
                replacement = replacement.replace(key, args[n])

        return head + replacement + tail

    def pull_out_newcommands(self, line):
        newcommands = ['\\newcommand*', '\\newcommand',
                       '\\renewcommand*', '\\renewcommand',
                       '\\DeclareMathOperator*', '\\DeclareMathOperator']
        operator = '{\\operatorname%s{%s}}'
        for cmd in newcommands:
            while self.contains_keyword(line, cmd):
                line, args = self.cut_out_command(line, cmd)
                replacement = args['replacement']
                if cmd == '\\DeclareMathOperator*':
                    args['replacement'] = operator % ('*', replacement)
                if cmd == '\\DeclareMathOperator':
                    args['replacement'] = operator % ('', replacement)
                self.macros[args['command name']] = args['replacement']
                if args['number of args'] == '':
                    self.macro_args[args['command name']] = (0, False, '')
                else:
                    self.macro_args[args['command name']] = (
                        int(args['number of args']),
                        args['optional'],
                        args['optional arg']
                    )
        return line

    def pull_out_newtheorems(self, line):
        newtheorems = ['\\newtheorem*', '\\newtheorem']
        for cmd in newtheorems:
            while self.contains_keyword(line, cmd):
                line, args = self.cut_out_theorem(line, cmd)
                envname = args['command name']
                theorem = args['theorem name'].lower()
                self.theorems.append(theorem)
                begin = '\\begin{%s}'
                end = '\\end{%s}'
                self.macros[begin % envname] = begin % theorem
                self.macro_args[begin % envname] = (0, False, '')
                self.macros[end % envname] = end % theorem
                self.macro_args[end % envname] = (0, False, '')
        return line

    def cut_out_theorem(self, line, cmd):
        args = {}
        cmd_start = self.keyword_index(line, cmd)
        cmd_end = cmd_start + len(cmd)
        reduced_line = line[:cmd_start]
        tail = line[cmd_end:]
        tail, args['command name'] = self.cut_out_delimited(tail, '{', '}')
        tail, args['numbering'] = self.cut_out_delimited(tail, '[', ']')
        tail, args['theorem name'] = self.cut_out_delimited(tail, '{', '}')
        tail, args['number within'] = self.cut_out_delimited(tail, '[', ']')
        return reduced_line + tail, args

    def cut_out_command(self, line, cmd):
        args = {}
        cmd_start = self.keyword_index(line, cmd)
        cmd_end = cmd_start + len(cmd)
        reduced_line = line[:cmd_start]
        tail = line[cmd_end:]
        tail, args['command name'] = self.cut_out_command_name(tail)
        tail, args['number of args'] = self.cut_out_delimited(tail, '[', ']')
        if tail.startswith('['):
            args['optional'] = True
            tail, args['optional arg'] = self.cut_out_delimited(tail, '[', ']')
        else:
            args['optional'] = False
            args['optional arg'] = ''
        tail, args['replacement'] = self.cut_out_delimited(tail, '{', '}')
        return reduced_line + tail, args

    def cut_out_command_name(self, line):
        if line and line[0] == '{':
            _, position = self.find_matching_brackets(
                line, opening='{', closing='}')
            return line[position+1:], line[1:position]
        elif line and line[0] == '\\':
            for (n, c) in enumerate(line):
                if c in ['{', '[']:
                    return line[n:], line[:n]
        raise ValueError("Cannot parse the command name")

    def cut_out_delimited(self, line, opening, closing):
        if line.startswith(opening):
            _, position = self.find_matching_brackets(
                line, opening=opening, closing=closing)
            return line[position+len(closing):], line[len(opening):position]
        return line, ''

    def find_matching_brackets(self, line, opening, closing):
        depth = 0
        start = 0
        for n, c in enumerate(line):
            if (line[n:].startswith(opening) and
                    not line[n-1:n].startswith('\\')):
                if depth == 0:
                    start = n
                depth = depth + 1
            if (line[n:].startswith(closing) and
                    not line[n-1:n].startswith('\\')):
                depth = depth - 1
                if depth == 0:
                    return start, n
        raise ValueError("Cannot find matching bracket")

    def expand_macros_in_line(self, line):
        for cmd in self.macros:
            while self.contains_keyword(line, cmd):
                line = self.expand_once(line, cmd)
        return line

    def contains_keyword(self, line, keyword):
        return self.keyword_index(line, keyword) != -1

    def keyword_index(self, line, keyword):
        start = line.find(keyword)
        if keyword[-1] not in string.ascii_letters or start == -1:
            return start
        while True:
            next_char = start + len(keyword)
            c = line[next_char:next_char+1]
            if c == '' or c not in string.ascii_letters:
                return start
            next_match = line[start+1:].find(keyword)
            if next_match == -1:
                return -1
            start = start + 1 + next_match


def attempt_compile(filename, timeout):
    directory, name = os.path.split(filename)
    command = ['pdflatex', '-halt-on-error',
               '-output-directory', directory, name
               ]
    try:
        subprocess.run(command, timeout=timeout, check=True)
    except subprocess.CalledProcessError:
        return CompileCodes.ERROR
    except subprocess.TimeoutExpired:
        return CompileCodes.TIMEOUT
    return CompileCodes.OK


def create_xml_from_pdf(texdir):
    infile = os.path.join(texdir, 'test.pdf')
    outfile = os.path.join(texdir, 'test.xml')
    command = ['pdf2txt.py', '-o', outfile, infile]
    subprocess.run(command, timeout=3, check=True)
    tree = ET.parse(outfile)
    return tree.getroot()


def parse_into_tokens(latex_string):
    start_keyword = False
    in_keyword = False
    in_envname = False
    tokens = []
    current = ''
    for ch in latex_string:
        cycle = True
        while cycle:
            if start_keyword:
                cycle = False
                if ch in string.ascii_letters:
                    start_keyword = False
                    in_keyword = True
                    current = current + ch
                else:
                    start_keyword = False
                    in_keyword = False
                    tokens.append(current + ch)
                    current = ''
            elif in_keyword:
                if ch in string.ascii_letters:
                    current = current + ch
                    cycle = False
                else:
                    in_keyword = False
                    if current in ['\\begin', '\\end']:
                        in_envname = True
                        if ch != '{':
                            raise KeyError('%s not followed by {' % current)
                        current = current + ch
                        cycle = False
                    else:
                        tokens.append(current)
                        current = ''
                        cycle = True
            elif in_envname:
                cycle = False
                if ch == '}':
                    tokens.append(current + ch)
                    in_envname = False
                    current = ''
                else:
                    current = current + ch
            elif ch == '\\':
                cycle = False
                start_keyword = True
                current = ch
            else:
                cycle = False
                tokens.append(ch)
    return tokens


class CompileCodes(Enum):
    OK = 0
    ERROR = 1
    TIMEOUT = 2
