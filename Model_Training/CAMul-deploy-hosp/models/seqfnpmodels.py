from models.fnpmodels import (
    torch,
    nn,
    F,
    Normal,
    float_tensor,
    logitexp,
    SelfAttention,
    LatentAtten,
    sample_DAG,
    sample_Clique,
    sample_bipartite,
)
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, time_idx) -> torch.Tensor:
        """
        Args:
            time_idx: (seq_len)
        """
        x = self.pe[time_idx, :]  # (seq_len, d_model)
        return x


class PositionalDecoder(nn.Module):
    """
    AR Decoder with positional encoding input
    """

    def __init__(self, input_dim, output_dim, max_time=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim
        self.output_dim = output_dim
        self.max_time = max_time
        self.lstm = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.pos_enc = PositionalEncoding(self.input_dim, max_time)

    def forward(self, time_idx, hidden):
        """
        :param time_idx: (T)
        :param hidden: (batch_size, hidden_dim)
        :return: (batch_size, 1, output_dim)
        """
        pos_enc = self.pos_enc(time_idx)  # (T, input_dim)
        batch_size = hidden.shape[0]
        pos_enc = pos_enc.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (batch_size, T, input_dim)
        out, hidden = self.lstm(
            pos_enc, hidden.unsqueeze(0)
        )  # (batch_size, T, hidden_dim)
        out = self.linear(out)  # (batch_size, T, output_dim)
        return out


class RegressionSeqFNP(nn.Module):
    """
    Functional Neural Process for regression
    """

    def __init__(
        self,
        dim_x=1,
        dim_y=1,
        num_t=4,
        dim_h=50,
        transf_y=None,
        n_layers=1,
        use_plus=True,
        num_M=100,
        dim_u=1,
        dim_z=1,
        fb_z=0.0,
        use_DAG=True,
        add_atten=False,
    ):
        """
        :param dim_x: Dimensionality of the input
        :param dim_y: Dimensionality of the output
        :param dim_h: Dimensionality of the hidden layers
        :param transf_y: Transformation of the output (e.g. standardization)
        :param n_layers: How many hidden layers to use
        :param use_plus: Whether to use the FNP+
        :param num_M: How many points exist in the training set that are not part of the reference set
        :param dim_u: Dimensionality of the latents in the embedding space
        :param dim_z: Dimensionality of the  latents that summarize the parents
        :param fb_z: How many free bits do we allow for the latent variable z
        """
        super(RegressionSeqFNP, self).__init__()

        self.num_M = num_M
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_t = num_t
        self.dim_h = dim_h
        self.dim_u = dim_u
        self.dim_z = dim_z
        self.use_plus = use_plus
        self.fb_z = fb_z
        self.transf_y = transf_y
        self.use_DAG = use_DAG
        self.add_atten = add_atten
        # normalizes the graph such that inner products correspond to averages of the parents
        self.norm_graph = lambda x: x / (torch.sum(x, 1, keepdim=True) + 1e-8)

        self.register_buffer("lambda_z", float_tensor(1).fill_(1e-8))

        # function that assigns the edge probabilities in the graph
        self.pairwise_g_logscale = nn.Parameter(
            float_tensor(1).fill_(math.log(math.sqrt(self.dim_u)))
        )
        self.pairwise_g = lambda x: logitexp(
            -0.5
            * torch.sum(
                torch.pow(x[:, self.dim_u :] - x[:, 0 : self.dim_u], 2), 1, keepdim=True
            )
            / self.pairwise_g_logscale.exp()
        ).view(x.size(0), 1)
        # transformation of the input

        init = [nn.Linear(dim_x, self.dim_h), nn.ReLU()]
        for i in range(n_layers - 1):
            init += [nn.Linear(self.dim_h, self.dim_h), nn.ReLU()]
        self.cond_trans = nn.Sequential(*init)
        # p(u|x)
        self.p_u = nn.Linear(self.dim_h, 2 * self.dim_u)
        # q(z|x)
        self.q_z = nn.Linear(self.dim_h, 2 * self.dim_z)
        # for p(z|A, XR, yR)

        # p(y|z) or p(y|z, u)
        # TODO: Add for sR input
        self.atten_ref = SelfAttention(self.dim_x)
        self.output = PositionalDecoder(
            self.dim_z + self.dim_x
            if not self.use_plus
            else self.dim_z + self.dim_u + self.dim_x,
            2 * dim_y,
            self.num_t,
        )
        if self.add_atten:
            self.atten_layer = LatentAtten(self.dim_h)

    def forward(self, XR, XM, yM, time_idx, kl_anneal=1.0):
        # sR = self.atten_ref(XR).mean(dim=0)
        sR = XR.mean(dim=0)
        X_all = torch.cat([XR, XM], dim=0)
        H_all = self.cond_trans(X_all)

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        # get G
        if self.use_DAG:
            G = sample_DAG(u[0 : XR.size(0)], self.pairwise_g, training=self.training)
        else:
            G = sample_Clique(
                u[0 : XR.size(0)], self.pairwise_g, training=self.training
            )

        # get A
        A = sample_bipartite(
            u[XR.size(0) :], u[0 : XR.size(0)], self.pairwise_g, training=self.training
        )
        if self.add_atten:
            HR, HM = H_all[0 : XR.size(0)], H_all[XR.size(0) :]
            atten = self.atten_layer(HM, HR)
            A = A * atten

        # get Z
        qz_mean_all, qz_logscale_all = torch.split(self.q_z(H_all), self.dim_z, 1)
        qz = Normal(qz_mean_all, qz_logscale_all)
        z = qz.rsample()
        pz_mean_all = torch.mm(
            self.norm_graph(torch.cat([G, A], dim=0)),
            qz_mean_all[0 : XR.size(0)],
        )
        pz_logscale_all = torch.mm(
            self.norm_graph(torch.cat([G, A], dim=0)),
            qz_logscale_all[0 : XR.size(0)],
        )

        pz = Normal(pz_mean_all, pz_logscale_all)

        pqz_all = pz.log_prob(z) - qz.log_prob(z)

        # apply free bits for the latent z
        if self.fb_z > 0:
            log_qpz = -torch.sum(pqz_all)

            if self.training:
                if log_qpz.item() > self.fb_z * z.size(0) * z.size(1) * (1 + 0.05):
                    self.lambda_z = torch.clamp(
                        self.lambda_z * (1 + 0.1), min=1e-8, max=1.0
                    )
                elif log_qpz.item() < self.fb_z * z.size(0) * z.size(1):
                    self.lambda_z = torch.clamp(
                        self.lambda_z * (1 - 0.1), min=1e-8, max=1.0
                    )

            log_pqz_M = self.lambda_z * torch.sum(pqz_all[XR.size(0) :])

        else:
            log_pqz_M = torch.sum(pqz_all[XR.size(0) :])

        final_rep = z if not self.use_plus else torch.cat([z, u], dim=1)
        sR = sR.repeat(final_rep.shape[0], 1)
        final_rep = torch.cat([sR, final_rep], dim=-1)

        mean_y, logstd_y = torch.split(self.output(time_idx, final_rep), 1, dim=-1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        _, mean_yM = mean_y[0 : XR.size(0)], mean_y[XR.size(0) :]
        _, logstd_yM = logstd_y[0 : XR.size(0)], logstd_y[XR.size(0) :]

        # logp(M|S)
        pyM = Normal(mean_yM, logstd_yM)
        log_pyM = torch.sum(pyM.log_prob(yM))

        obj_M = (log_pyM + log_pqz_M) / float(XM.size(0))

        obj = obj_M

        loss = -obj

        return loss, mean_y, logstd_y

    def predict(self, x_new, XR, time_idx, sample=True):
        # sR = self.atten_ref(XR).mean(dim=0)
        sR = XR.mean(dim=0)
        H_all = self.cond_trans(torch.cat([XR, x_new], 0))

        # get U
        pu_mean_all, pu_logscale_all = torch.split(self.p_u(H_all), self.dim_u, dim=1)
        pu = Normal(pu_mean_all, pu_logscale_all)
        u = pu.rsample()

        A = sample_bipartite(
            u[XR.size(0) :], u[0 : XR.size(0)], self.pairwise_g, training=False
        )

        if self.add_atten:
            HR, HM = H_all[0 : XR.size(0)], H_all[XR.size(0) :]
            atten = self.atten_layer(HM, HR)
            A = A * atten

        pz_mean_all, pz_logscale_all = torch.split(
            self.q_z(H_all[0 : XR.size(0)]), self.dim_z, 1
        )
        pz_mean_all = torch.mm(self.norm_graph(A), pz_mean_all)
        pz_logscale_all = torch.mm(self.norm_graph(A), pz_logscale_all)
        pz = Normal(pz_mean_all, pz_logscale_all)

        z = pz.rsample()
        final_rep = z if not self.use_plus else torch.cat([z, u[XR.size(0) :]], dim=1)
        sR = sR.repeat(final_rep.shape[0], 1)
        final_rep = torch.cat([sR, final_rep], dim=-1)

        mean_y, logstd_y = torch.split(self.output(time_idx, final_rep), 1, dim=-1)
        logstd_y = torch.log(0.1 + 0.9 * F.softplus(logstd_y))

        init_y = Normal(mean_y, logstd_y)
        if sample:
            y_new_i = init_y.sample()
        else:
            y_new_i = mean_y

        y_pred = y_new_i

        if self.transf_y is not None:
            if torch.cuda.is_available():
                y_pred = self.transf_y.inverse_transform(y_pred.cpu().data.numpy())
            else:
                y_pred = self.transf_y.inverse_transform(y_pred.data.numpy())

        return y_pred, mean_y, logstd_y, u[XR.size(0) :], u[: XR.size(0)], init_y, A
