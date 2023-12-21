import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, LayerNorm, global_max_pool


class AttentionLayer(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(AttentionLayer, self).__init__()
        self.attention = torch.nn.Linear(2 * input_features, output_features)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, edge_out, edge_attr):
        combined_features = torch.cat((edge_out, edge_attr), dim=1)
        attention_scores = self.attention(combined_features)
        attention_weights = self.softmax(attention_scores)
        attended_features = attention_weights * combined_features
        return attended_features.sum(dim=1).unsqueeze(1)


class GNNModel(torch.nn.Module):
    def __init__(
        self,
        input_features=9,
        hidden_features=9,
        output_features=1,
        num_layers=2,
        dropout=0.5,
        leaky_or_tanh=True,
        use_gat=True,
        use_attention=True,
    ):
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        ConvLayer = GATConv if use_gat else GCNConv

        self.ln1 = torch.nn.LayerNorm(input_features)

        self.conv_layers = torch.nn.ModuleList(
            [ConvLayer(input_features, hidden_features)]
        )
        self.norm_layers = torch.nn.ModuleList([LayerNorm(hidden_features)])

        for _ in range(1, num_layers):
            self.conv_layers.append(ConvLayer(hidden_features, hidden_features))
            self.norm_layers.append(LayerNorm(hidden_features))

        self.leaky_or_tanh_func = F.leaky_relu if leaky_or_tanh else torch.tanh
        self.dropout = torch.nn.Dropout(p=dropout)

        self.use_attention = use_attention
        if self.use_attention:
            self.attention_layer = AttentionLayer(output_features, output_features)

        self.ln2 = torch.nn.LayerNorm(2 * hidden_features)
        self.edge_transform = torch.nn.Linear(2 * hidden_features, hidden_features)
        self.fc = torch.nn.Linear(hidden_features, output_features)

    def forward(self, data, training=False):
        x, edge_index, edge_attr = (
            data.x,
            data.edge_index,
            data.edge_attr,
        )
        x = self.ln1(x)

        for i in range(self.num_layers):
            identity = x
            x = self.conv_layers[i](x, edge_index, edge_attr)
            if i != 0:
                x = self.norm_layers[i](x) + identity
            x = self.leaky_or_tanh_func(x)

            if training and i != self.num_layers - 1:
                x = self.dropout(x)

        edge_features = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=1)
        edge_features = self.ln2(edge_features)

        edge_out = self.leaky_or_tanh_func(self.edge_transform(edge_features))
        edge_out = self.dropout(edge_out)
        edge_out = self.fc(edge_out)

        if self.use_attention:
            edge_out = self.attention_layer(edge_out, edge_attr) + edge_out

        return edge_out
