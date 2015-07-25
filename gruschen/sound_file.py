import random
import struct
import unittest
import wave

import numpy as np


class SoundFileTest(unittest.TestCase):
    maxDiff = None

    def randomsignal(self):
        return Signal([random.random() for _ in range(1024)], 1024)

    def test_save_and_load(self):
        signal = self.randomsignal()
        signal[0] = 1
        filename = "elutenexvetxvnle.wav"
        save(filename, signal)
        retrieved_signal = load(filename)
        self.assertEqual(len(signal), len(retrieved_signal))
        self.assertEqual(signal.sampleRate, retrieved_signal.sampleRate)
        scaled_retrieved = signal[0] / retrieved_signal[0] * retrieved_signal
        for i in range(len(signal)):
            # this test fails right now, because the loss of the conversion
            # while saving / loading the soundfile is too big
            self.assertAlmostEqual(signal[i], scaled_retrieved[i])


exportNumberType = np.int16


def load(filename):
    wavfile = wave.open(filename, "r")
    nchannels, sampwidth, samplerate, nframes, _, _ = wavfile.getparams()
    frames = wavfile.readframes(nframes * nchannels)
    sampwidth_to_character = {1: "b", 2: "h", 4: "i", 8: "q"}
    filedata = struct.unpack("<{}{}".format(nframes * nchannels,
                                            sampwidth_to_character[sampwidth]),
                             frames)

    return np.array(filedata[0::nchannels], dtype=np.float), samplerate


def save(filename, signal, samplerate):
    signal = _convert_number_type(signal, exportNumberType)
    _write_to_wave_file(filename, signal, samplerate)


def _convert_number_type(signal, number_type):
    return list(np.array([x.real for x in signal]) *
                (min(np.iinfo(number_type).max,
                     abs(np.iinfo(number_type).min)) - 1))


def _write_to_wave_file(filename, signal, samplerate):
    wavfile = wave.open(filename, mode='wb')
    wavfile.setparams((1, exportNumberType(1).itemsize, samplerate,
                      len(signal), 'NONE', 'not compressed'))
    wavfile.writeframes(np.array(signal, dtype=exportNumberType))
    wavfile.close()
