import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torchmetrics import MeanSquaredError, F1Score
from src.GNLayer import GNLayer
from src.utils import tensor_to_list

class MLPBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


class GNBlock(nn.Module):
    def __init__(
        self,
        conv_type,
        in_channels,
        hidden_channels,
        out_channels,
        edge_dim=0,
        use_norm=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'GCNConv':
          assert edge_dim <= 1, 'GCNConv only supports edge_dim <= 1'
          self.conv = gnn.GCNConv(in_channels, out_channels)
        elif conv_type == 'ChebConv':
          assert edge_dim <= 1, 'ChebConv only supports edge_dim <= 1'
          self.conv = gnn.ChebConv(in_channels, out_channels, K=1)
        elif conv_type == 'GATConv':
          self.conv = gnn.GATv2Conv(in_channels, out_channels, edge_dim=edge_dim)
        elif conv_type == 'GINConv':
          if edge_dim == 0:
            self.conv = gnn.GINConv(MLPBlock(in_channels, hidden_channels, out_channels))
          else:
            self.conv = gnn.GINEConv(MLPBlock(in_channels, hidden_channels, out_channels), edge_dim=edge_dim)
        elif conv_type == 'GraphConv':
          assert edge_dim <= 1, 'GraphConv only supports edge_dim <= 1'
          self.conv = gnn.GraphConv(in_channels, out_channels)
        elif conv_type == 'NNConv':
          self.conv = GNLayer(in_channels, hidden_channels, out_channels, edge_dim=edge_dim)
        else:
          raise ValueError(f'conv_type must be one of "GCNConv", "ChebConv", "GATConv", "GINConv", "GraphConv", "NNConv", but got {conv_type}')
        self.act = nn.LeakyReLU()
        self.use_norm = use_norm
        if use_norm:
          self.norm = gnn.BatchNorm(out_channels, track_running_stats=True)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr)
        x = self.act(x)
        if self.use_norm and x.size(0) > 1:
          x = self.norm(x)
        return x


class GNN(nn.Module):
    def __init__(
        self,
        conv_type,
        task_type,
        learning_rate,
        in_channels,
        num_layers,
        hidden_channels,
        out_channels,
        edge_dim=0,
        use_norm=False,
        class_weights=None,
      ):
      super().__init__()

      self.conv_type = conv_type
      self.task_type = task_type # 'classification' or 'regression'
      self.learning_rate = learning_rate
      self.in_channels = in_channels
      self.num_layers = num_layers
      self.hidden_channels = hidden_channels
      self.out_channels = out_channels
      self.edge_dim = edge_dim
      self.use_norm = use_norm
      self.task_type = task_type

      self.convs = nn.ModuleList()
      self.convs.append(GNBlock(conv_type, in_channels, hidden_channels, hidden_channels, edge_dim, use_norm))
      for _ in range(num_layers - 1):
        self.convs.append(GNBlock(conv_type, hidden_channels, hidden_channels, hidden_channels, edge_dim, use_norm))
      self.head = MLPBlock(hidden_channels, hidden_channels, out_channels)
      self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
      if task_type == 'classification':
        self.register_buffer("class_weights", class_weights)
        self.criterion = lambda logits, target, reduction='mean': F.cross_entropy(logits, target, weight=self.class_weights, reduction=reduction)
        self.metric = F1Score(task='multiclass', num_classes=out_channels, average='weighted')
      elif task_type == 'regression':
        self.criterion = F.mse_loss
        self.metric = MeanSquaredError(squared=False)
      else:
        raise ValueError('Task type must be either "classification" or "regression"')
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      if self.device != torch.device('cuda'):
        print('WARNING: GPU not available. Using CPU instead.')
      self.to(self.device)

    def embed(self, data):
      x = data.x
      for conv in self.convs:
        x = conv(x, data.edge_index, data.edge_attr)
      x = gnn.global_mean_pool(x, data.batch, size=data.batch_size)
      return x

    def forward(self, data):
      x = self.embed(data)
      x = self.head(x)
      return x

    def train_batch(self, loader):
      self.train()
      metrics = {
         'loss': 0.0,
         'metric': 0.0,
      }
      for data in loader:
        data = data.to(self.device)
        if self.task_type == 'regression':
          data.y = data.y.unsqueeze(1)
        self.optimizer.zero_grad()
        logits = self(data)
        loss = self.criterion(logits, data.y)
        loss.backward()
        self.optimizer.step()
        metrics['loss'] += loss.item()
        metrics['metric'] += self.metric(logits, data.y).item()
      for metric_name in metrics:
        metrics[metric_name] /= len(loader)
      return metrics

    @torch.no_grad()
    def evaluate_batch(self, loader):
      self.eval()
      metrics = {
         'loss': 0.0,
         'metric': 0.0,
      }
      for data in loader:
        data = data.to(self.device)
        if self.task_type == 'regression':
          data.y = data.y.unsqueeze(1)
        logits = self(data)
        loss = self.criterion(logits, data.y)
        metrics['loss'] += loss.item()
        metrics['metric'] += self.metric(logits, data.y).item()
      for metric_name in metrics:
        metrics[metric_name] /= len(loader)
      return metrics

    @torch.no_grad()
    def predict_batch(self, loader):
      self.eval()
      x_embs = []
      y_preds = []
      y_trues = []
      for data in loader:
        data = data.to(self.device)
        if self.task_type == 'regression':
          data.y = data.y.unsqueeze(1)
        x_emb = self.embed(data)
        y_pred = self.head(x_emb)
        if self.task_type == 'classification':
          y_pred = F.softmax(y_pred, dim=1)
        x_embs += tensor_to_list(x_emb)
        y_preds += tensor_to_list(y_pred)
        y_trues += tensor_to_list(data.y)
      return x_embs, y_preds, y_trues
