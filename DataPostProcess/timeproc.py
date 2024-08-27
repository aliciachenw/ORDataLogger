import numpy as np

def timesync(timeframe1, timeframe2, threshold=1):
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


def get_sequence_list(timeframe):
    record_flags = timeframe[:,-1]

    sequence_list = []

    start = 0
    for i in range(1, len(record_flags)):
        if record_flags[i] == 0 and record_flags[i-1] == 1:
            sequence_list.append([start, i-1])
        elif record_flags[i] == 1 and record_flags[i-1] == 0:
            start = i
    
    if record_flags[-1] == 1 and start == 0:
        sequence_list.append([start, len(record_flags)-1])

    return sequence_list