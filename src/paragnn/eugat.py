"""Edge-Updated Graph Attention Network.

Adapted from IL-PCSR (Paul et al., EMNLP 2025).
Original: IL-PCSR/utils/EUGATConv.py + IL-PCSR/utils/eugatgnn.py
Based on SCENE paper: https://arxiv.org/pdf/2301.03512.pdf
"""
import math

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.utils.internal import expand_as_pair
from dgl._ffi.base import DGLError
from dgl.ops import edge_softmax


class EUGATConv(nn.Module):
    """Graph attention layer with edge features from SCENE."""

    def __init__(
        self,
        in_feats,
        edge_feats,
        out_feats,
        out_edge_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        allow_zero_in_degree=False,
        bias=False,
    ):
        super(EUGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer("bias", None)

        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            self.res_fc_edge = nn.Linear(edge_feats, num_heads * out_edge_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)

        self._edge_feats = edge_feats
        self.fc_edge = nn.Linear(edge_feats, out_feats * num_heads, bias=False)
        self.attn_edge = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))

        # Edge update layers
        self.fc_ni = nn.Linear(self._in_src_feats, out_edge_feats * num_heads, bias=False)
        self.fc_nj = nn.Linear(self._in_src_feats, out_edge_feats * num_heads, bias=False)
        self.fc_fij = nn.Linear(edge_feats, out_edge_feats * num_heads, bias=False)
        self._out_edge_feats = out_edge_feats

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_edge, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, edge_feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph. "
                        "Add self-loops via `g = dgl.add_self_loop(g)`."
                    )

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            # Edge feature transform
            n_edges = edge_feat.shape[:-1]
            feat_edge = self.fc_edge(edge_feat).view(*n_edges, self._num_heads, self._out_feats)
            graph.edata["ft_edge"] = feat_edge

            # Attention computation
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            ee = (feat_edge * self.attn_edge).sum(dim=-1).unsqueeze(-1)
            graph.edata["ee"] = ee

            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            graph.apply_edges(fn.u_add_v("el", "er", "e_tmp"))
            graph.edata["e"] = graph.edata["e_tmp"] + graph.edata["ee"]

            # Combined edge features (source node + edge)
            graph.apply_edges(fn.u_add_e("ft", "ft_edge", "ft_combined"))

            e = self.leaky_relu(graph.edata.pop("e"))
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            graph.edata["m_combined"] = graph.edata["ft_combined"] * graph.edata["a"]
            graph.update_all(fn.copy_e("m_combined", "m"), fn.sum("m", "ft"))

            rst = graph.dstdata["ft"]

            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats
                )
            if self.activation:
                rst = self.activation(rst)

            # Edge update
            graph.srcdata.update({"f_ni": feat_src})
            graph.dstdata.update({"f_nj": feat_dst})
            graph.apply_edges(fn.u_add_v("f_ni", "f_nj", "f_tmp"))
            f_out = graph.edata.pop("f_tmp") + feat_edge
            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = nn.functional.leaky_relu(f_out)
            f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)

            return rst, f_out


class EUGATGNN(nn.Module):
    """2-layer Edge-Updated Graph Attention Network with residual connections."""

    def __init__(self, in_dim, h_dim, out_dim, dropout, num_head):
        super(EUGATGNN, self).__init__()
        self.hidden_size = h_dim
        self.in_dim = in_dim
        self.EUGATConv1 = EUGATConv(
            in_feats=in_dim, edge_feats=in_dim, out_feats=out_dim,
            out_edge_feats=out_dim, num_heads=num_head, allow_zero_in_degree=True,
        )
        self.EUGATConv2 = EUGATConv(
            in_feats=in_dim, edge_feats=in_dim, out_feats=out_dim,
            out_edge_feats=out_dim, num_heads=num_head, allow_zero_in_degree=True,
        )
        self.embedding_dropout1 = nn.Dropout(dropout)
        self.embedding_dropout2 = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size if self.hidden_size else self.in_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, g, node_feats, edge_feats):
        # Layer 1
        h = self.EUGATConv1(g, node_feats, edge_feats)
        h_0 = th.squeeze(h[0])  # node features
        h_1 = th.squeeze(h[1])  # edge features
        h_0 = self.embedding_dropout1(h_0)
        h_1 = self.embedding_dropout2(h_1)
        h_0 = F.relu(h_0) + node_feats  # residual
        h_1 = F.relu(h_1) + edge_feats  # residual

        # Layer 2
        h = self.EUGATConv2(g, h_0, h_1)
        h_0 = F.relu(h_0)
        h = th.squeeze(h[0]) + node_feats  # residual from input

        return h
