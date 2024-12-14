import torch
import torch.nn as nn
import torch.nn.functional as F
from sparse_softmax import Sparsemax
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import (
    softmax,
    dense_to_sparse,
    coalesce,
    degree,
    add_self_loops
)
from torch_scatter import scatter
from torch_sparse import spspmm, coalesce
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn.pool.connect.filter_edges import filter_adj


class TwoHopNeighborhood(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        num_nodes = data.num_nodes
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index_2hop, edge_weight_2hop = spspmm(
            edge_index, edge_weight,
            edge_index, edge_weight,
            num_nodes, num_nodes, num_nodes
        )
        mask = edge_index_2hop[0] != edge_index_2hop[1]
        edge_index_2hop = edge_index_2hop[:, mask]
        edge_ids = edge_index[0] * num_nodes + edge_index[1]
        edge_2hop_ids = edge_index_2hop[0] * num_nodes + edge_index_2hop[1]
        edge_ids_sorted = edge_ids.sort()[0]
        edge_2hop_ids_sorted, inverse_indices = edge_2hop_ids.sort()
        idx = torch.searchsorted(edge_ids_sorted, edge_2hop_ids_sorted)
        idx = torch.clamp(idx, max=edge_ids_sorted.size(0) - 1)
        mask = edge_ids_sorted[idx] != edge_2hop_ids_sorted
        new_edge_indices = inverse_indices[mask]
        new_edge_index = edge_index_2hop[:, new_edge_indices]
        if new_edge_index.size(1) > 0:
            # Handle edge attributes if they exist
            if data.edge_attr is not None:
                new_edge_attr = torch.zeros(
                    new_edge_index.size(1),
                    dtype=data.edge_attr.dtype,
                    device=data.edge_attr.device
                )
                data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)
            # Concatenate the new edges
            data.edge_index = torch.cat([edge_index, new_edge_index], dim=1)
            data.edge_index, data.edge_attr = coalesce(
                data.edge_index, data.edge_attr, num_nodes, num_nodes
            )
        else:
            data.edge_index = edge_index

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)



class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False):
        super(NodeInformationScore, self).__init__(aggr='add')
        self.improved = improved
        self.cached = cached
        self._cached_edge_index = None
        self._cached_norm = None

    def forward(self, x, edge_index, edge_weight=None):
        if self.cached and self._cached_edge_index is not None:
            edge_index, norm = self._cached_edge_index, self._cached_norm
        else:
            num_nodes = x.size(0)
            if edge_weight is None:
                edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)

            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
            )

            row, col = edge_index
            deg = degree(row, num_nodes, dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

            if self.cached:
                self._cached_edge_index = edge_index
                self._cached_norm = norm

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class HGPSLPool(nn.Module):
    def __init__(
        self, in_channels, pool_ratio=0.8, lambda_coef=1.0,
        negative_slope=0.2, apply_sparsemax=False, structure_learning=True, use_sampling=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pool_ratio = pool_ratio
        self.lambda_coef = lambda_coef
        self.negative_slope = negative_slope
        self.apply_sparsemax = apply_sparsemax
        self.structure_learning = structure_learning
        self.use_sampling = use_sampling

        self.attention_weights = Parameter(torch.Tensor(1, in_channels * 2))
        nn.init.xavier_uniform_(self.attention_weights)
        self.sparse_attention = Sparsemax()
        self.expand_neighbors = TwoHopNeighborhood()
        self.node_info_score = NodeInformationScore()

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        batch = self._initialize_batch(batch, x)
        node_scores = self._compute_node_scores(x, edge_index, edge_attr)
        selected_nodes = self._select_nodes(node_scores, batch)

        x_pool, batch_pool, edge_index_pool, edge_attr_pool = self._pool_features(
            x, edge_index, edge_attr, batch, selected_nodes
        )

        if not self.structure_learning:
            return x_pool, edge_index_pool, edge_attr_pool, batch_pool

        return self._apply_structure_learning(
            x_pool, edge_index_pool, edge_attr_pool, batch_pool, node_scores, selected_nodes
        )

    def _initialize_batch(self, batch, x):
        return batch if batch is not None else x.new_zeros(x.size(0), dtype=torch.long)

    def _compute_node_scores(self, x, edge_index, edge_attr):
        info_scores = self.node_info_score(x, edge_index, edge_attr)
        return torch.abs(info_scores).sum(dim=1)

    def _select_nodes(self, node_scores, batch):
        return topk(node_scores, self.pool_ratio, batch)

    def _pool_features(self, x, edge_index, edge_attr, batch, selected_nodes):
        x_pool = x[selected_nodes]
        batch_pool = batch[selected_nodes]
        edge_index_pool, edge_attr_pool = filter_adj(edge_index, edge_attr, selected_nodes, x.size(0))
        return x_pool, batch_pool, edge_index_pool, edge_attr_pool

    def _apply_structure_learning(self, x_pool, edge_index_pool, edge_attr_pool, batch_pool, node_scores, selected_nodes):
        if self.use_sampling:
            return self._sample_neighbors_based_structure(
                x_pool, edge_index_pool, edge_attr_pool, selected_nodes, node_scores.size(0)
            )
        else:
            return self._full_graph_based_structure(
                x_pool, edge_index_pool, edge_attr_pool, batch_pool
            )

    def _sample_neighbors_based_structure(self, x_pool, edge_index, edge_attr, selected_nodes, num_nodes):
        data = Data(x=x_pool, edge_index=edge_index, edge_attr=edge_attr)
        khop = 3
        for _ in range(khop-1):  # Expand two more hops
            data = self.expand_neighbors(data)

        edge_index_new, edge_attr_new = self._generate_new_graph_weights(
            data.edge_index, data.edge_attr, num_nodes, x_pool
        )
        return x_pool, edge_index_new, edge_attr_new, None

    def _full_graph_based_structure(self, x_pool, edge_index_pool, edge_attr_pool, batch_pool):
        num_nodes_per_batch = scatter(
            torch.ones_like(batch_pool), batch_pool, dim=0, reduce='sum'
        )
        adjacency_matrix = self._build_block_diagonal(num_nodes_per_batch, x_pool.device)
        edge_index_new, _ = dense_to_sparse(adjacency_matrix)

        edge_index_new, edge_attr_new = self._generate_new_graph_weights(
            edge_index_new, edge_attr_pool, x_pool.size(0), x_pool
        )
        return x_pool, edge_index_new, edge_attr_new, batch_pool

    def _build_block_diagonal(self, num_nodes_per_batch, device):
        blocks = [torch.ones(n, n, device=device) for n in num_nodes_per_batch]
        return torch.block_diag(*blocks)

    def _generate_new_graph_weights(self, edge_index, edge_attr, num_nodes, x_pool):
        row, col = edge_index
        edge_features = torch.cat([x_pool[row], x_pool[col]], dim=-1)
        weights = (edge_features * self.attention_weights).sum(dim=-1)
        weights = F.leaky_relu(weights, self.negative_slope)

        if edge_attr is not None:
            weights += edge_attr * self.lambda_coef

        if self.apply_sparsemax:
            edge_attr_new = self.sparse_attention(weights, row)
        else:
            edge_attr_new = softmax(weights, row, num_nodes=num_nodes)

        return edge_index, edge_attr_new

