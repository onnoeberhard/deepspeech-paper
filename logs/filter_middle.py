"""Some testing logs include memory leak warnings that make them hard to read.
This script gets rid of them. (The warnings, not the memory leaks..)"""
import re

with open('middle.txt') as fi, \
     open('middle_clean.txt', 'w') as fo:
    for line in fi:
        if match := re.search(r'Test epoch \| Steps: \d+ \| Elapsed Time: \d+:\d+:\d+', line):
            fo.write(match.group(0) + '\n')
