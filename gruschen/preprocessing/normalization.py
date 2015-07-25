
def normalize(signal):
    return signal / max(max(signal), abs(min(signal)))
