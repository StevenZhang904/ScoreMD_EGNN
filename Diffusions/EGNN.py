from copy import deepcopy
from torch import nn
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
import numpy as np
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim//2) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(
        self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), 
        residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            # print(source.shape, "source.shape", target.shape, "target.shape", radial.shape, "radial.shape")
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            # print(x.shape, "x.shape", agg.shape, "agg.shape")
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        
        edge_feat = self.edge_model(h[row],h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, 
        hidden_channels, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=3, 
        residual=True, attention=False, normalize=False, tanh=False, 
        max_atom_type=2, cutoff=0, max_num_neighbors=32, **kwargs
    ):
        '''
        :param max_atom_type: Number of features for 'h' at the input
        :param hidden_channels: Number of hidden features
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.max_atom_type = max_atom_type
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.type_embedding = nn.Embedding(max_atom_type, hidden_channels)

        self.t_emb = nn.Sequential(
            GaussianFourierProjection(embed_dim=hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            act_fn,
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Gaussian = GaussianFourierProjection(self.hidden_channels)
        # self.act_fn = act_fn
        # linear = nn.Linear(hidden_channels, hidden_channels)

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(
                self.hidden_channels, self.hidden_channels, self.hidden_channels, edges_in_d=in_edge_nf,
                act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh))
        
        self.position_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels*2),
            nn.SiLU(), 
            nn.Linear(hidden_channels*2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 18)
        )

    def forward(self, z, pos, disp, batch, t,  edge_index=None, edge_attr=None):


        h = self.type_embedding(z)
        # print(h.shape)

        temb = torch.stack((t, t, t, t, t, t), axis = 1).reshape(-1, 1)
        temb = self.t_emb(temb)

        h += temb
        h = torch.squeeze(h)
        x = deepcopy(disp)
        if edge_index is None:
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                loop=False,
                max_num_neighbors=self.max_num_neighbors + 1,
            )
        # print(edge_index.shape, "edge_index.shape")
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edge_index, x, edge_attr = edge_attr)
            # h += temb
            # h = torch.squeeze(h)

        # out = scatter(h, batch, dim=0, reduce='sum')
        # # print(out.shape, "out.shape")
        # out = self.position_head(out)
        noise = x - disp
        # print(out.shape, "out.shape")
        return noise 



def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)