import numpy as np

def timesync(timeframe1, timeframe2, threshold=1e-2):
    """
    Function to synchronize two timeframes
    """
    matches = []
    # find the closest match
    for i, t1 in enumerate(timeframe1):
        j = np.argmin(np.abs(timeframe2 - t1))
        if np.abs(timeframe2[j] - t1) < threshold:
            matches.append([i, j])
    matches = np.array(matches)
    return matches


def extract_sequence():
    pass