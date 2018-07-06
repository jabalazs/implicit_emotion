#!/usr/bin/env python3

import re
import sys
from unicode_codes_py3 import EMOJI_UNICODE

emoji_pattern = '[' + ''.join(EMOJI_UNICODE.values()) + ']'
emoji_regexp = re.compile(emoji_pattern)


def retag_entry(line):
    if line == '\n':
        return '\n'

    token, tag = line.split()
    if emoji_regexp.match(token):
        tag = "E"

    return ' '.join([token, tag]) + '\n'


if __name__ == "__main__":
    for line in sys.stdin:
        sys.stdout.write(retag_entry(line))
