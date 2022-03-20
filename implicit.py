import torch
import torch.nn.functional as F
import numpy as np

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)

        return torch.linalg.norm(
            sample_points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        diff = torch.abs(sample_points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)


sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


from torch.nn import Parameter, init


class LinearWithRepeat(torch.nn.Module):
    """
    reference the pytorch3d project nerf
    """

    def __init__(self,in_features: int,out_features: int,bias: bool = True,device=None,dtype=None,):
        """
        Copied from torch.nn.Linear.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Copied from torch.nn.Linear.
        """
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


# TODO (3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        n_layers = cfg.n_layers_xyz
        input_dim = cfg.n_hidden_neurons_xyz
        hidden_dim = input_dim
        skip_dim = embedding_dim_xyz
        append_xyz = cfg.append_xyz
        hidden_dir_dim = cfg.n_hidden_neurons_dir

        self.mlp = MLPWithInputSkips(
            n_layers,
            embedding_dim_xyz,
            input_dim,
            skip_dim,
            hidden_dim,
            input_skips=append_xyz,
        )

        self.density_layer = torch.nn.Linear(hidden_dim, 1)
        self.density_layer.bias.data[:] = 0.0

        self.color_layer_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.color_layer_2 = torch.nn.Sequential(
            LinearWithRepeat(hidden_dim + embedding_dim_dir, hidden_dir_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(hidden_dir_dim, 3),
            torch.nn.Sigmoid(),
        )
        print(self.mlp)

    def get_colors(self, features, rays_directions):
        rays_directions_normed = rays_directions

        intermedia = self.color_layer_1(features)
        rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        return self.color_layer_2((intermedia, rays_embedding))

    def get_densities(self, features):
        return torch.relu(self.density_layer(features))

    def forward(self, ray_bundle: RayBundle):
        embeds_xyz = self.harmonic_embedding_xyz(ray_bundle.sample_points)
        features = self.mlp(embeds_xyz, embeds_xyz)
        colors = self.get_colors(features, ray_bundle.directions)

        densities = self.get_densities(features)
        
        out = {'density':densities, 'feature':colors}
        return out

# TODO (3.1): Implement NeRF MLP
class NeuralRadianceFieldBase(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        print("Warming base model !!!!!!!!!!!")

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        n_layers = cfg.n_layers_xyz
        input_dim = cfg.n_hidden_neurons_xyz
        hidden_dim = input_dim
        skip_dim = embedding_dim_xyz
        append_xyz = cfg.append_xyz
        hidden_dir_dim = cfg.n_hidden_neurons_dir

        self.mlp = MLPWithInputSkips(
            n_layers,
            embedding_dim_xyz,
            input_dim,
            skip_dim,
            hidden_dim,
            input_skips=append_xyz,
        )

        self.density_layer = torch.nn.Linear(hidden_dim, 1)
        self.density_layer.bias.data[:] = 0.0

        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dir_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(hidden_dir_dim, 3),
            torch.nn.Sigmoid(),
        )
        print(self.mlp)

    def get_colors(self, features):
        return self.color_layer(features)

    def get_densities(self, features):
        return torch.relu(self.density_layer(features))

    def forward(self, ray_bundle: RayBundle):
        embeds_xyz = self.harmonic_embedding_xyz(ray_bundle.sample_points)
        features = self.mlp(embeds_xyz, embeds_xyz)
        colors = self.get_colors(features)

        densities = self.get_densities(features)
        
        out = {'density':densities, 'feature':colors}
        return out


volume_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'nerf_raw': NeuralRadianceFieldBase,
}