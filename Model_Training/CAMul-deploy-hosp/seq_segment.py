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
    _, n_timesteps = sequences.shape
    bkps = np.linspace(0, n_timesteps, segments + 1, dtype=int)
    ans = []
    for j in range(len(bkps) - 1):
        ans.append(sequences[:, bkps[j] : bkps[j + 1]])
    ans = np.concatenate(ans, axis=0)
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
        bkps = rpt.Binseg(model=dist).fit_predict(sequences[i,:], n_bkps=segments)
        if len(bkps) == 1:
            seqs.extend(uniform_segment(sequences, segments))
            continue
        for j in range(len(bkps) - 1):
            seqs.append(sequences[:, bkps[j] : bkps[j + 1]])
    return seqs
