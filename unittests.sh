#!/bin/bash
filenames=$(find gruschen/ -name "*.py" | tr '\n' ' ')
PYTHONPATH=. /usr/bin/env python3 -m unittest $* ${filenames}
