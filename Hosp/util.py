import torch
from torch.nn.utils.rnn import pad_sequence


class TorchStandardScaler:
    def fit(self, x, device):
        x = torch.tensor(x).float().to(device)
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
        # self.mean = torch.mean(x, keepdim=True)
        # self.std = torch.std(x, keepdim=True, unbiased=False)

    def transform(self, x):
        if torch.is_tensor(x):
            x -= self.mean
            x /= (self.std + 1e-7)
        else:
            x -= self.mean.cpu().numpy()
            x /= (self.std + 1e-7).cpu().numpy()
        return x

    def fit_transform(self, x, device):
        self.fit(x, device)
        return self.transform(x)

    def inverse_transform(self, x):
        x *= self.std
        x += self.mean
        return x


def create_time_seq(no_sequences, sequence_length):
    """
    Creates windows of fixed size
    """
    # convert to small sequences for training, starting with length 10
    seqs = []
    # starts at sequence_length and goes until the end
    for idx in range(no_sequences):
        # Sequences
        seqs.append(torch.arange(idx,idx+sequence_length))
    seqs = pad_sequence(seqs,batch_first=True).type(torch.float)
    return seqs