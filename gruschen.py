#!/usr/bin/python3

import argparse
import glob
import os
import re
import sys
import unittest

import signal_comparison


class Recording:
    def __init__(self, name, text, filename):
        self.name = name
        self.text = text
        self.filename = filename

    def __repr__(self):
        return u"[{}] {} ({})".format(self.text, self.name, self.filename)


def run_tests():
    unittest.main()


def getargs():
    parser = argparse.ArgumentParser(usage="""Cluster voice files.""")
    voice_group = parser.add_mutually_exclusive_group(required=True)
    voice_group.add_argument(
        "--voice-dir",
        help="A directory from which to take all .wav files.",
        action="append",
        default=[],
    )
    voice_group.add_argument(
        "--voice-files",
        help="A list of files to cluster.",
        nargs="+",
        default=[],
    )
    return parser.parse_args()


def to_recordings(directories, files):
    def _get_subject_word(fpath):
        name, extension = os.path.splitext(os.path.basename(fpath))
        match = re.match(r"(\D+)\d+", name)
        if match:
            return match.group(1)
        return name

    recordings = []
    recordings += [
        Recording(
            os.path.basename(fpath),
            _get_subject_word(fpath),
            fpath,
        )
        for fpath in files
    ]
    recordings += [
        Recording(
            os.path.basename(fpath),
            _get_subject_word(fpath),
            fpath,
        )
        for directory in directories
        for fpath in glob.glob(os.path.join(directory, "*.wav"))
    ]
    print(recordings)
    return recordings


def main(args):
    args = getargs()
    recordings = to_recordings(args.voice_dir, args.voice_files)
    distances = signal_comparison.compare_files(recordings)
    print(distances)
    return 0


if __name__ == '__main__':
    sys.exit(main(getargs()))
