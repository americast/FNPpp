import torch
import torch.nn as nn
import math

class TransformerAttn(nn.Module):
    """
    Module that calculates self-attention weights using transformer like attention
    """
    def __init__(self, dim_in=40, value_dim=40, key_dim=40):
        """
        param dim_in: Dimensionality of input sequence
        param value_dim: Dimension of value transform
        param key_dim: Dimension of key transform
        """
        super(TransformerAttn, self).__init__()
        self.value_layer = nn.Linear(dim_in, value_dim)
        self.query_layer = nn.Linear(dim_in, value_dim)
        self.key_layer = nn.Linear(dim_in, key_dim)

    def forward(self, seq):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.softmax(weights, -1)
        return (weights @ keys).transpose(1, 0)

    def forward_mask(self, seq, mask):
        """
        param seq: Sequence in dimension [Seq len, Batch, Hidden size]
        param mask: Sequence in dimesnion [Seq len, Batch, Hidden size]
            Mask to unused time series
        """
        seq_in = seq.transpose(0, 1)
        value = self.value_layer(seq_in)
        query = self.query_layer(seq_in)
        keys = self.key_layer(seq_in)
        weights = (value @ query.transpose(1, 2)) / math.sqrt(seq.shape[-1])
        weights = torch.exp(weights)
        weights = (weights.transpose(1, 2) * mask.transpose(1, 0)).transpose(1, 2)
        weights = weights / (weights.sum(-1, keepdim=True))
        return (weights @ keys).transpose(1, 0) * mask



class EmbedAttenSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention module
    """
    def __init__(
        self,dim_seq_in=5, rnn_out=40, dim_out=50, dim_metadata=48, 
        n_layers =1, bidirectional=False, attn=TransformerAttn, dropout=0
        ):
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param rnn_out: output dimension for rnn
        """
        super(EmbedAttenSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.dim_metadata = dim_metadata
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.attn_layer = attn(self.rnn_out, self.rnn_out, self.rnn_out)
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out + self.dim_metadata, out_features=self.dim_out
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward(self, seqs, metadata):
        """
        param seqs: Sequence in dimension [Seq len, Batch, Hidden size]
            Specifically, Seq len = Batch + min_window_size - 1
        param metadata: One-hot encoding for region information 
        """
        # Take last output from GRU 
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer(latent_seqs).sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out

    def forward_mask(self, seqs, mask, metadata):
        """
        param seqs: Sequence in dimension [Seq len, Batch, Hidden size]
            Specifically, Seq len = Batch + min_window_size - 1
        param mask: Sequence in dimesnion [Seq len, Batch, Hidden size]
            Mask to unused time series
        param metadata: One-hot encoding for region information 
        """
        latent_seqs = self.rnn(seqs)[0]
        latent_seqs = self.attn_layer.forward_mask(latent_seqs, mask)
        latent_seqs = latent_seqs.sum(0)
        out = self.out_layer(torch.cat([latent_seqs, metadata], dim=1))
        return out



class EncoderModules(nn.Module):
    def __init__(self, dim_seq_in=5, device=None):
        super(EncoderModules, self).__init__()

        self.mods = EmbedAttenSeq(
                dim_seq_in=dim_seq_in, rnn_out=64,  # divides by 2 if bidirectional
                dim_out=32, n_layers=2, bidirectional=True
            ).to(device)



class DecodeSeq(nn.Module):
    """
    Module to embed a sequence. Adds Attention modul
    """
    def __init__(
        self, dim_seq_in=5, dim_metadata=3, rnn_out=40, dim_out=5, 
        n_layers=1, bidirectional=False, dropout=0.0
        ):
        """
        param dim_seq_in: Dimensionality of input vector (no. of age groups)
        param dim_out: Dimensionality of output vector
        param dim_metadata: Dimensions of metadata for all sequences
        param rnn_out: output dimension for rnn
        """
        super(DecodeSeq, self).__init__()

        self.dim_seq_in = dim_seq_in
        self.dim_metadata = dim_metadata
        self.rnn_out = rnn_out
        self.dim_out = dim_out
        self.bidirectional = bidirectional

        self.act_fcn = nn.Tanh()

        self.rnn = nn.GRU(
            input_size=self.dim_seq_in,
            hidden_size=self.rnn_out // 2 if self.bidirectional else self.rnn_out,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
        )
        self.out_layer = [
            nn.Linear(
                in_features=self.rnn_out, out_features=self.dim_out
            ),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        self.out_layer = nn.Sequential(*self.out_layer)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.out_layer.apply(init_weights)

    def forward(self, Hi_data, hidden):
        # Hi_data is scaled time
        inputs = Hi_data.transpose(1,0)
        if self.bidirectional:
            h0 = hidden.expand(2,-1,-1).contiguous()
        else:
            h0 = hidden.unsqueeze(0)
        # Take last output from GRU
        latent_seqs = self.rnn(inputs, h0)[0]
        latent_seqs = latent_seqs.transpose(1,0)
        latent_seqs = self.out_layer(latent_seqs)
        return latent_seqs




class DecoderModules(nn.Module):
    def __init__(self, device=None):
        super(DecoderModules, self).__init__()

        self.mods = DecodeSeq(
                dim_seq_in=1, rnn_out=64, # divides by 2 if bidirectional
                dim_out=20, n_layers=1, bidirectional=True
            ).to(device)



class OutputModules(nn.Module):
    def __init__(self, out_dim=5, device=None):
        super(OutputModules, self).__init__()

        out_layer_width = 20
        out_layer =  [
            nn.Linear(
                in_features=out_layer_width, out_features=2*out_layer_width
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=2*out_layer_width, out_features=2*out_layer_width
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=2*out_layer_width, out_features=out_layer_width
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=out_layer_width, out_features=out_dim
            ),
        ]

        self.mods = nn.Sequential(*out_layer).to(device)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.mods.apply(init_weights)
