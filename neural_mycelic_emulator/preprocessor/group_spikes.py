import numpy as np
from typing import List

__all__ = [
    "group_spikes",
]

def group_spikes(spike_mask: np.ndarray, theta: int | None = None) -> List[List[int]]:
    """Group spike indices into words separated by long gaps.

    Parameters
    ----------
    spike_mask : np.ndarray[bool]
        Boolean array where True denotes a spike.
    theta : int | None
        Gap (in samples) above which a new word starts. If None, uses median inter-spike interval.

    Returns
    -------
    List[List[int]]
        List of words, each a list of spike indices.
    """
    indices = np.nonzero(spike_mask)[0]
    if len(indices) == 0:
        return []

    if theta is None:
        intervals = np.diff(indices)
        theta = int(np.median(intervals)) if len(intervals) else 1

    words = [[indices[0]]]
    for idx in indices[1:]:
        if idx - words[-1][-1] > theta:
            words.append([idx])
        else:
            words[-1].append(idx)
    return words 