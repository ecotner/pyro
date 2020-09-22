
class Z2Decoder(nn.Module):
    def __init__(self, z1_size, y_size, z2_size, hidden_sizes):
        super.__init__()
        layers = []
        sizes = [z1_size + y_size] + hidden_sizes + [z2_size + z2_size]
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        layers.pop(-1)
        self.fc = nn.Sequential(*layers)

    def forward(self, z1, y):
        z1_y = torch.cat([z1, y], dim=-1)
        loc_scale = self.fc(z1_y)
        loc, scale = loc_scale.reshape(loc_scale.shape[:-1] + (2, -1)).unbind(-2)
        scale = nn.functional.softplus(scale)
        return loc, scale


class SCANVI(nn.Module):
    def __init__(self, num_labels, l_loc, l_scale):
        assert isinstance(num_labels, int) and num_labels > 1
        self.num_labels = num_labels

        self.latent_dim = 10

        assert isinstance(l_loc, float)
        self.l_loc = l_loc

        assert isinstance(l_scale, float) and l_scale > 0
        self.l_scale = l_scale

        super().__init__()

        self.z2_decoder = Z2Decoder(z1_size=self.latent_dim,
                                    y_size=self.num_labels,
                                    z2_size=self.latent_dim,
                                    hidden_sizes=[100, 100])
        self.x_decoder = XDecoder(TODO)
        self.z2yl_encoder = Z2YLEncoder(TODO)
        self.z1_encoder = Z1Encoder(TODO)

    def model(self, x, y=None):
        pyro.module("scanvi", self)
        with pyro.plate("batch", len(xs)):
            z1 = pyro.sample("z1", dist.Normal(0, 1).expand([self.latent_dim]).to_event(1))
            y = pyro.sample("y", dist.OneHotCategorical(logits=torch.zeros(self.num_labels))

            z2_loc, z2_scale = self.z2_decoder(z1, y)
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            l = pyro.sample("l", dist.LogNormal(self.l_loc, self.l_scale))

            gate, total_count, logits = self.x_decoder(z2)
            x = pyro.sample("x", dist.ZeroInflatedNegativeBinomial(
                gate,
                total_count,
                logits,
            ), obs=x)

    def guide(self, x, y=None):
        pyro.module("scanvi", self)
        with pyro.plate("batch", len(xs)):
            z2_loc, z2_scale, y_logits, l_loc, l_scale = self.z2yl_encoder(x)
            pyro.sample("l", dist.LogNormal(l_loc, l_scale))
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            y_dist = dist.OneHotCategorical(logits=y_logits)
            if y is None:
                y = pyro.sample(y_dist)
            else:
                # Add a classification loss term.
                classification_loss = y_dist.log_prob(y)
                pyro.factor("classification_loss", -classification_loss)

            z1_loc, z1_scale = self.z1_encoder(y, z2)
            pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))


def train(x, y):
    scanvi = SCANVI(TODO)
    optim = ClippedAdam(TODO)
    guide = config_enumerate(scanvi.guide, expand=True)
    loss = SVI(scanvi.model, guide, optim, TraceEnum_ELBO())
    dataloader = TODO
    # TODO alternate between supervised and unsupervised batches
    # TODO KL annealing via poutine.scale


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.4.0')
