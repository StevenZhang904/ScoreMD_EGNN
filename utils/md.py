import numpy as np
import torch
from torch import nn

BOX_SIZE = 25.0

def gaussian_smearing_(distances, offset, widths, centered=False):
    if not centered:
        # Compute width of Gaussians (using an overlap of 1 STDDEV)
        # widths = offset[1] - offset[0]
        coeff = -0.5 / torch.pow(widths, 2)
        diff = distances - offset
    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # If centered Gaussians are requested, don't substract anything
        diff = distances
    del distances

    # Compute and return Gaussians
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    del diff
    
    return gauss


@torch.no_grad()
def pairwise_distance_norm(x, box_size, mask_self=False, cached_mask=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = \sqrt{||x[i,:]-y[j,:]||^2}
    '''
    dist_all = 0.
    for dim in range(x.shape[1]):
        x_norm = (x[:, 0] ** 2).view(-1, 1)
        y_t = x[:, 0].view(1, -1)
        y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x[:, 0].view(-1, 1), y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        # Ensure diagonal is zero if x=y
        del x_norm, y_norm, y_t
        dist -= torch.diag(dist.diag())
        dist = torch.clamp(dist, 0.0, np.inf)
        dist[dist != dist] = 0.
        dist = dist.view(-1)

        # dist = torch.remainder(dist + 0.5 * box_size, float(box_size)) - 0.5 * float(box_size)
        dist_mat_mask = dist > (box_size**2 / 4)
        dist[dist_mat_mask] = dist[dist_mat_mask] + box_size**2 -\
                              2.0 * box_size * torch.sqrt(dist[dist_mat_mask]) * torch.sign(dist[dist_mat_mask])

        del dist_mat_mask
        if dim != 2:
            x = x[:, 1:]
        dist_all += dist
        del dist
        torch.cuda.empty_cache()
    dist = torch.sqrt_(dist_all)
    if mask_self:
        # if cached_mask is None:
        #     self_mask = np.array([i*x.shape[0] + j for i in range(x.shape[0]) for j in range(x.shape[0]) if i == j])
        #     mask_array = np.ones(x.shape[0]*x.shape[0], dtype=bool)
        #     mask_array[self_mask] = False
        # else:
        #     mask_array = cached_mask
        dist = dist[dist > 0.]
    return dist, None


def pair_distance(pos: torch.Tensor, box_size, mask_self=False, return_norm=False, cached_mask=None):
    # [[0, 1, 2, ,3, 4 ...], [0, 1, ...],...]      [[0, 0, 0, 0, 0, ...], [1, 1, 1, ...],...]
    # print("shape of pos:", pos.shape)
    dist_mat = pos[None, :, :] - pos[:, None, :]
    # print("shape of dist_mat:", dist_mat.shape)
    dist_mat = torch.remainder(dist_mat + 0.5 * box_size, box_size) - 0.5 * box_size
    # print("shape of dist_mat after remainder:", dist_mat.shape)
    dist_mat = dist_mat.view(-1, pos.size(1))
    if mask_self:
        if cached_mask is None:
            self_mask = np.array([i*pos.shape[0] + j for i in range(pos.shape[0]) for j in range(pos.shape[0]) if i == j])
            mask_array = np.ones(pos.shape[0]*pos.shape[0], dtype=bool)
            mask_array[self_mask] = 0
        else:
            mask_array = cached_mask
        dist_mat = dist_mat[mask_array]
    if return_norm:
        return dist_mat.norm(dim=1), mask_array
    return dist_mat


def pair_distance_two_system(pos1: torch.Tensor, pos2: torch.Tensor, box_size):
    # pos1 and pos2 should in same shape
    # [[0, 1, 2, ,3, 4 ...], [0, 1, ...],...]      [[0, 0, 0, 0, 0, ...], [1, 1, 1, ...],...]
    dist_mat = pos1[None, :, :] - pos2[:, None, :]
    # dist_mat_mask_right = dist_mat > box_size / 2
    # dist_mat_mask_left = dist_mat < -box_size / 2
    #
    # dist_mat[dist_mat_mask_right] = dist_mat[dist_mat_mask_right] - box_size
    # dist_mat[dist_mat_mask_left] = dist_mat[dist_mat_mask_left] + box_size
    dist_mat = torch.remainder(dist_mat+0.5*box_size, box_size) - 0.5*box_size
    return dist_mat.view(-1, pos1.size(1))


def get_neighbor(pos: torch.Tensor, r_cutoff, box_size, return_dist=True,
                 predefined_mask=None, bond_type=None):

    if isinstance(pos, np.ndarray):
        if torch.cuda.is_available():
            pos = torch.from_numpy(pos).cuda()
            if bond_type is not None:
                bond_type = torch.from_numpy(bond_type).cuda()

    with torch.no_grad():
        distance = pair_distance(pos, box_size)
        # print("shape of distance:", distance.shape)
        distance_norm = torch.norm(distance, dim=1)  # [pos.size(0) * pos.size(0), 1]
        edge_idx_1 = torch.cat([torch.arange(pos.size(0)) for _ in range(pos.size(0))], dim=0).to(pos.device)
        edge_idx_2 = torch.cat([torch.LongTensor(pos.size(0)).fill_(i) for i in range(pos.size(0))], dim=0).to(pos.device)

        if predefined_mask is not None:
            mask = (distance_norm.view(-1) <= r_cutoff) & predefined_mask & ~(edge_idx_1 == edge_idx_2)
        else:
            mask = (distance_norm.view(-1) <= r_cutoff) & ~(edge_idx_1 == edge_idx_2)

        masked_bond_type = None
        if bond_type is not None:
            masked_bond_type = bond_type[mask]
        edge_idx_1 = edge_idx_1[mask].view(1, -1)
        edge_idx_2 = edge_idx_2[mask].view(1, -1)

        edge_idx = torch.cat((edge_idx_1, edge_idx_2), dim=0)
        distance = distance[mask]
        distance_norm = distance_norm[mask]

    if return_dist:
        return edge_idx, distance, distance_norm, masked_bond_type
    else:
        return edge_idx, masked_bond_type


class GaussianSmearing(nn.Module):
    """
    Wrapper class of gaussian_smearing function. Places a predefined number of Gaussian functions within the
    specified limits.
    sample struct dictionary:
        struct = {'start': 0.0, 'stop':5.0, 'n_gaussians': 32, 'centered': False, 'trainable': False}
    Args:
        start (float): Center of first Gaussian.
        stop (float): Center of last Gaussian.
        n_gaussians (int): Total number of Gaussian functions.
        centered (bool):  if this flag is chosen, Gaussians are centered at the origin and the
            offsets are used to provide their widths (used e.g. for angular functions).
            Default is False.
        trainable (bool): If set to True, widths and positions of Gaussians are adjusted during training. Default
            is False.
    """
    def __init__(self, start, stop, n_gaussians, width=None, centered=False, trainable=False):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        if width is None:
            widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        else:
            widths = torch.FloatTensor(width * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer('width', widths)
            self.register_buffer('offsets', offset)
        self.centered = centered

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.
        Returns:
            torch.Tensor: Tensor of convolved distances.
        """
        result = gaussian_smearing_(
            distances, self.offsets, self.width,  centered=self.centered
        )
        return result


class RDF(nn.Module):
    def __init__(self, nbins=400, r_range=(0, BOX_SIZE/2), width=None):
        super(RDF, self).__init__()
        start = r_range[0]
        end = r_range[1]
        self.bins = torch.linspace(start, end, nbins + 1)
        self.smear = GaussianSmearing(
            start=start,
            stop=self.bins[-1],
            n_gaussians=nbins,
            width=width,
            trainable=False
        )
        self.cut_off = end - start
        self.box_size = self.cut_off * 2
        # compute volume differential
        self.vol_bins = 4 * np.pi / 3 * (self.bins[1:] ** 3 - self.bins[:-1] ** 3)
        self.nbins = nbins
        self.cached_mask = None

    def forward(self, pos: torch.Tensor, divided=False, batch_size=int(1e6)):
        if self.cached_mask is None:
            pair_dist, self_mask = pairwise_distance_norm(pos, self.box_size, mask_self=True)
            self.cached_mask = self_mask
        else:
            pair_dist, _ = pairwise_distance_norm(pos, self.box_size, mask_self=True, cached_mask=self.cached_mask)
        pair_dist = pair_dist.detach()
        if not divided:
            count = self.smear(pair_dist.view(-1).squeeze()[..., None]).sum(0)
        else:
            count = torch.zeros((self.nbins)).cuda()
            for b in range(pair_dist.shape[0] // batch_size + 1):
                end = b*batch_size + batch_size
                if b*batch_size + batch_size >= pair_dist.shape[0]:
                    end = -1
                count += self.smear(pair_dist[b*batch_size:end].view(-1).squeeze()[..., None]).sum(0)
            del pair_dist
            count = count
        norm = count.sum()  # normalization factor for histogram
        count = count / norm  # normalize

        V = (4 / 3) * np.pi * (self.cut_off ** 3)
        rdf = count.to(self.vol_bins.device) / (self.vol_bins / V)

        return count, self.bins, rdf


class RDF2Sys(nn.Module):
    def __init__(self, nbins=400, r_range=(0, BOX_SIZE/2), width=None):
        super(RDF2Sys, self).__init__()
        start = r_range[0]
        end = r_range[1]
        self.bins = torch.linspace(start, end, nbins + 1)
        self.smear = GaussianSmearing(
            start=start,
            stop=self.bins[-1],
            n_gaussians=nbins,
            width=width,
            trainable=False
        )
        self.cut_off = end - start
        self.box_size = self.cut_off * 2
        # compute volume differential
        self.vol_bins = 4 * np.pi / 3 * (self.bins[1:] ** 3 - self.bins[:-1] ** 3)
        self.nbins = nbins

    def forward(self, pos1: torch.Tensor, pos2: torch.Tensor):
        pair_dist = torch.norm(pair_distance_two_system(pos1, pos2, self.box_size), dim=1)

        pair_dist = pair_dist.detach()
        pair_dist = pair_dist[pair_dist > 1.]
        count = self.smear(pair_dist.view(-1).squeeze()[..., None]).sum(0)
        norm = count.sum()  # normalization factor for histogram
        count = count / norm  # normalize

        V = (4 / 3) * np.pi * (self.cut_off ** 3)
        rdf = count.to(self.vol_bins.device) / (self.vol_bins / V)

        return count, self.bins, rdf
