from typing import List, Tuple
import ruptures as rpt
import numpy as np


def uniform_segment(sequences: np.ndarray, segments: int = 1):
    """Segments a sequence of sequences into n_bkps segments.

    Args:
        sequences (np.ndarray): A sequence of sequences. [n_sequences, n_timesteps]
        n_bkps (int, optional): Number of breakpoints. Defaults to 1.

    Returns:
        np.ndarray: A sequence of breakpoints. [n_sequences*n_bkps, n_timesteps//n_bkps]
    """
    n_sequences, n_timesteps = sequences.shape
    bkps = np.linspace(0, n_timesteps, segments + 1, dtype=int)
    ans = []
    for i in range(n_sequences):
        for j in range(len(bkps) - 1):
            ans.append(sequences[i, bkps[j] : bkps[j + 1]])
    return ans


def pelt_segment(sequences: np.ndarray, segments: int = 1):
    """Segments a sequence of sequences into n_bkps segments.

    Args:
        sequences (np.ndarray): A sequence of sequences. [n_sequences, n_timesteps]
        n_bkps (int, optional): Number of breakpoints. Defaults to 1.

    Returns:
        np.ndarray: A sequence of breakpoints. [n_sequences*n_bkps, n_timesteps//n_bkps]
    """
    n_sequences, n_timesteps = sequences.shape
    bkps = rpt.Pelt(model="rbf").fit_predict(sequences, pen=segments)
    seqs = []
    for i in range(n_sequences):
        if len(bkps) == 1:
            seqs.append(uniform_segment(sequences, segments))
            continue
        ans = []
        for j in range(len(bkps) - 1):
            ans.append(sequences[:, bkps[j] : bkps[j + 1]])
        ans = np.concatenate(ans, axis=0)
        seqs.append(ans)
    return np.concatenate(seqs, axis=0)


def binary_segment(sequences: np.ndarray, segments: int = 4, dist="l1"):
    """Segments a sequence of sequences into n_bkps segments.

    Args:
        sequences (np.ndarray): A sequence of sequences. [n_sequences, n_timesteps]
        n_bkps (int, optional): Number of breakpoints. Defaults to 1.

    Returns:
        np.ndarray: A sequence of breakpoints. [n_sequences*n_bkps, n_timesteps//n_bkps]
    """
    n_sequences, n_timesteps = sequences.shape
    seqs = []
    for i in range(n_sequences):
        bkps = rpt.Binseg(model=dist).fit_predict(sequences[i, :], n_bkps=segments - 1)
        if bkps[0] != 0:
            bkps = [0] + bkps
        print(bkps)
        if len(bkps) == 1:
            print("Warning: No breakpoints found. Using uniform segmentation.")
            seqs.extend(uniform_segment(sequences[None, i], segments))
            continue
        for j in range(len(bkps) - 1):
            seqs.append(sequences[i, bkps[j] : bkps[j + 1]])
    return seqs


def combine_seqs_masks(seqs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combines multiple 1D sequences of different lens to a
    2D array and also computes masks (1 where sequence ends)

    Args:
        seqs (list): list of 1D sequences

    Returns:
        np.ndarray: 2D array of sequences
        np.ndarray: 2D array of masks
    """
    seq_lens = [len(seq) for seq in seqs]
    max_seq_len = max(seq_lens)
    np_seqs = np.zeros((len(seqs), max_seq_len))
    masks = np.zeros((len(seqs), max_seq_len))
    for i, seq in enumerate(seqs):
        np_seqs[i, : seq_lens[i]] = seq
        masks[i, seq_lens[i] - 1] = 1.0
    return np_seqs, masks
