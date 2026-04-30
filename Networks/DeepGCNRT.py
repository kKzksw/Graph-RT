import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.utils import softmax


class GCNLayerWithEdge(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        residual=True,
        output_norm="none",
        dropout=0.1,
        update_func="no_relu",
    ):
        super().__init__()

        self.mlp = nn.Linear(in_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.update_func = update_func
        self.activation = nn.ReLU()

        if update_func == "relu_eps_beta":
            self.eps = 1e-7
            self.beta = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        if output_norm == "batch_norm":
            self.norm = nn.BatchNorm1d(out_feats)
        elif output_norm == "layer_norm":
            self.norm = nn.LayerNorm(out_feats)
        elif output_norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported output_norm: {output_norm}")

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        messages = x[src] + edge_attr

        if self.update_func == "relu_eps_beta":
            messages = F.relu(messages) + self.eps
            attention = softmax(messages * self.beta, dst, num_nodes=x.size(0))
        elif self.update_func == "no_relu":
            attention = softmax(messages, dst, num_nodes=x.size(0))
        elif self.update_func == "relu":
            messages = F.relu(messages)
            attention = softmax(messages, dst, num_nodes=x.size(0))
        else:
            raise ValueError(f"Unsupported update_func: {self.update_func}")

        messages = messages * attention
        new_x = x.new_zeros(x.size(0), messages.size(1))
        new_x.index_add_(0, dst, messages)

        new_x = self.mlp(new_x)
        new_x = self.activation(new_x)
        new_x = self.dropout(new_x)

        if self.residual:
            new_x = new_x + x

        new_x = self.norm(new_x)
        return new_x


class AttentiveGraphReadout(nn.Module):
    def __init__(self, hidden_dim, num_timesteps=2, dropout=0.1):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.mol_conv = GATConv(
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            add_self_loops=False,
            negative_slope=0.01,
        )
        self.mol_conv.explain = False
        self.mol_gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, batch):
        out = global_add_pool(x, batch).relu_()
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        for _ in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        return out


class Network(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim=200,
        num_layers=16,
        readout_steps=2,
        dropout=0.1,
        norm="none",
        update_func="no_relu",
    ):
        super().__init__()

        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList(
            GCNLayerWithEdge(
                hidden_dim,
                hidden_dim,
                residual=True,
                output_norm=norm,
                dropout=dropout,
                update_func=update_func,
            )
            for _ in range(num_layers)
        )

        self.readout = AttentiveGraphReadout(
            hidden_dim,
            num_timesteps=readout_steps,
            dropout=dropout,
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        for layer in self.gnn_layers:
            x = layer(x, data.edge_index, edge_attr)

        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        out = self.readout(x, batch)
        out = self.out(out)
        return out.view(-1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
