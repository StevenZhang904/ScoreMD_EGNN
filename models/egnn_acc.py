from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch_scatter import scatter

from models.pbc_utils import radius_graph_pbc, get_pbc_distances


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), residual=True, coords_weight=1.0, attention=False, clamp=False, normalize=False, tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
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
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        # coord += agg*self.coords_weight
        # coord = coord + agg * self.coords_weight
        return agg * self.coords_weight

    def coord2radial(self, coord_diff):
        # row, col = edge_index
        # coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    # def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
    def forward(self, data, h, coord, edge_index, edge_attr=None, node_attr=None):
        __, coord_diff, __ = get_pbc_distances(
            data, 
            coord,
            edge_index,
            data.cell,
            data.cell_offsets,
            data.natoms,
        )
        # row, col = edge_index
        row, col = edge_index[0], edge_index[1]
        radial, coord_diff = self.coord2radial(coord_diff)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class E_GCL_vel(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """


    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), residual=True, coords_weight=1.0, attention=False, normalize=False, tanh=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim, act_fn=act_fn, residual=residual, coords_weight=coords_weight, attention=attention, normalize=normalize, tanh=tanh)
        self.normalize = normalize
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))
        self.coord_mlp_disp = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))

    def forward(self, data, h, coord, acc, edge_index, edge_attr=None, node_attr=None):
        # print('prev coord:', torch.min(coord), torch.max(coord))
        coord = torch.remainder(coord, data.cell[0, 0, 0])
        # print('new coord:', torch.min(coord), torch.max(coord))
        # print('box size:', data.cell[0, 0, 0])
        __, coord_diff, __ = get_pbc_distances(
            coord,
            edge_index,
            data.cell,
            data.cell_offsets,
            data.natoms,
        )

        # row, col = edge_index
        row, col = edge_index[0], edge_index[1]
        radial, coord_diff = self.coord2radial(coord_diff)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        # coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        # coord += self.coord_mlp_vel(h) * acc * 0.01
        disp = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        disp += self.coord_mlp_vel(h) * acc
        coord += disp
        
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, disp, edge_attr


class EGNN(nn.Module):
    def __init__(self, 
        in_node_nf, hidden_nf, 
        in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4, 
        residual=True, attention=False, normalize=False, tanh=False,
        auto_grad=False, cutoff=5.0, max_nbr=32,
    ):
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
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
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.auto_grad = auto_grad
        self.cutoff = cutoff
        self.max_nbr = max_nbr

        self.feat_emb = nn.Embedding(in_node_nf, hidden_nf)
        self.t_emb = nn.Sequential(
            GaussianFourierProjection(embed_dim=hidden_nf),
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_vel(hidden_nf, hidden_nf, hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn, 
            nn.Linear(hidden_nf, 1)
        )

    def _build_graph(self, data):
        bond_index = data.edge_index

        edge_index, cell_offsets, _, _ = radius_graph_pbc(
            data, self.cutoff, self.max_nbr
        )
        data.edge_index = edge_index
        data.cell_offsets = cell_offsets

        # include bond information to edge_attr
        edge_attr = torch.zeros(edge_index.shape[1], 1)
        bond_set = set()
        for i in range(bond_index.shape[1]):
            bond_set.add((bond_index[0, i].item(), bond_index[1, i].item()))
        for i in range(edge_attr.shape[0]):
            if (edge_index[0, i].item(), edge_index[1, i].item()) in bond_set:
                edge_attr[i, 0] = 1.0
        data.edge_attr = edge_attr.to(data.pos.device)

        return data

    def forward(self, data, t):
        data = self._build_graph(data)

        if self.auto_grad:
            data.pos.requires_grad_(True)
        
        temb = self.t_emb(t)
        h = self.feat_emb(data.feat)
        h += temb

        displacement = torch.zeros_like(data.pos).to(data.pos.device)

        for i in range(0, self.n_layers):
            if i == 0:
                h, x, disp, _ = self._modules["gcl_%d" % i](
                    data, h, data.pos, data.acc, data.edge_index, data.edge_attr
                )
            elif i == self.n_layers - 1:
                h, x, disp, _ = self._modules["gcl_%d" % i](
                    data, h, x, torch.zeros_like(x).to(x.device), data.edge_index, data.edge_attr
                )
            else:
                h, x, disp, _ = self._modules["gcl_%d" % i](
                    data, h, x, data.acc, data.edge_index, data.edge_attr
                )
            
            h += temb
            displacement += disp
        
        energy = scatter(h, data.batch, dim=0, reduce='add')
        energy = self.energy_head(energy)

        if self.auto_grad:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True
                )[0]
            )
        else:
            # forces = x - data.pos
            forces = displacement

        return energy, forces

    def load_pl_state_dict(self, state_dict):
        own_state = self.state_dict()
        for pl_name, param in state_dict.items():
            name = pl_name.replace('model.', '')
            if name not in own_state:
                print('Skipping parameter %s' % name)
                continue
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
                print('Loading parameter %s' % name)
            own_state[name].copy_(param)


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
