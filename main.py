import os
import os.path as osp

import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from torch.nn import Linear

from torch_geometric.data import TemporalData

from torch_geometric.logging import init_wandb, log
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
import numpy as np
import random
import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wikipedia') # wikipedia, reddit
parser.add_argument('--tensorboard', action='store_true', default=True)
args = parser.parse_args()
import time

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    
setup_seed(10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'FPGN-{args.dataset}')
if(args.tensorboard):
    writer = SummaryWriter(f'runs/FPGN-{args.dataset}')

root_path = osp.dirname(osp.realpath(__file__))
data = torch.load(osp.join(root_path, 'data', args.dataset+'.pt'))

data = data.to(device)

min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

train_data = TemporalData(src=data.src, dst=data.dst, t=data.t, msg=data.msg_core)
train_loader = TemporalDataLoader(train_data, batch_size=10000)

neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)

class WeightGenerator(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin1 = Linear(in_channels, in_channels)
        self.lin2 = Linear(in_channels, 1)

    def forward(self, inputs):
        return self.lin2(self.lin1(inputs).relu()).sigmoid() + 1

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).sigmoid()

memory_dim = time_dim = embedding_dim = 100
pos_src = data.train_pos_data[:,0]
pos_dst = data.train_pos_data[:,1]
neg_src = data.train_neg_data[:,0]
neg_dst = data.train_neg_data[:,1]

val_pos_src = data.val_pos_data[:,0]
val_pos_dst = data.val_pos_data[:,1]
val_neg_src = data.val_neg_data[:,0]
val_neg_dst = data.val_neg_data[:,1]

test_pos_src = data.test_pos_data[:,0]
test_pos_dst = data.test_pos_data[:,1]
test_neg_src = data.test_neg_data[:,0]
test_neg_dst = data.test_neg_data[:,1]

memory = TGNMemory(
    data.num_nodes,
    data.msg_core.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg_core.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg_core.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

get_pos_weight = WeightGenerator(in_channels=3).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()) | set(get_pos_weight.parameters()), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()

assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

def train():
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state() 
    neighbor_loader.reset_state() 
    optimizer.zero_grad()

    start_time = time.time()
    for batch in train_loader:
        batch = batch.to(device)

        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)

    n_id = torch.cat([pos_src, pos_dst, neg_src, neg_dst]).unique()
    n_id, edge_index, e_id = neighbor_loader(n_id)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)

    z, last_update = memory(n_id)
    z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
            data.msg_core[e_id].to(device))

    pos_out = link_pred(z[assoc[pos_src]], z[assoc[pos_dst]])
    neg_out = link_pred(z[assoc[neg_src]], z[assoc[neg_dst]])

    pos_weights = get_pos_weight(data.train_pos_attr)

    loss = (- pos_weights * torch.log(pos_out + 1e-9)).mean()
    loss += (- torch.log(1 - neg_out + 1e-9)).mean()
    loss.backward()
    optimizer.step()
    memory.detach()
    train_time = time.time()-start_time
    # scheduler.step()

    return loss, train_time

@torch.no_grad()
def val():
    memory.eval()
    gnn.eval()
    link_pred.eval()

    start_time = time.time()
    n_id = torch.cat([val_pos_src, val_pos_dst, val_neg_src, val_neg_dst]).unique()
    n_id, edge_index, e_id = neighbor_loader(n_id)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)

    z, last_update = memory(n_id)
    z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
            data.msg_core[e_id].to(device))

    pos_out = link_pred(z[assoc[val_pos_src]], z[assoc[val_pos_dst]])
    neg_out = link_pred(z[assoc[val_neg_src]], z[assoc[val_neg_dst]])

    out = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
    test_out = out > best_th
    val_time = time.time() - start_time
    y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)

    best_th = 0
    best_acc = 0
    for th in range(1, 100):
        th = th / 100
        acc = accuracy_score(y_true, out >= th)
        if(acc > best_acc):
            best_acc = acc
            best_th = th
    acc, f1, pre, recall = accuracy_score(y_true, out >= best_th), f1_score(y_true, out >= best_th), precision_score(y_true, out >= best_th), recall_score(y_true, out >= best_th)

    auc = roc_auc_score(y_true, out)
    ap = average_precision_score(y_true, out)

    return acc, f1, pre, recall, best_th, auc, ap, val_time

@torch.no_grad()
def test(best_th):
    memory.eval()
    gnn.eval()
    link_pred.eval()
    start_time = time.time()
    n_id = torch.cat([test_pos_src, test_pos_dst, test_neg_src, test_neg_dst]).unique()
    n_id, edge_index, e_id = neighbor_loader(n_id)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)

    z, last_update = memory(n_id)
    z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
            data.msg_core[e_id].to(device))

    pos_out = link_pred(z[assoc[test_pos_src]], z[assoc[test_pos_dst]])
    neg_out = link_pred(z[assoc[test_neg_src]], z[assoc[test_neg_dst]])

    out = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
    test_time = time.time()- start_time
    y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)

    acc, f1, pre, recall = accuracy_score(y_true, out >= best_th), f1_score(y_true, out >= best_th), precision_score(y_true, out >= best_th), recall_score(y_true, out >= best_th)

    auc = roc_auc_score(y_true, out)
    ap = average_precision_score(y_true, out)

    return acc, f1, pre, recall, auc, ap, test_time

best_acc = best_f1 = best_pre = best_recall = best_auc = final_test = best_epoch = 0
for epoch in range(1, 3001):
    loss, train_time = train()
    val_acc, val_f1, val_pre, val_recall, best_th, val_auc, val_ap, val_time = val()
    acc, f1, pre, recall, auc, ap, test_time = test(best_th)
    log(Epoch=epoch, Loss=loss, th=best_th, val_auc = val_auc, val_ap = val_ap, val_acc = val_acc, val_f1 = val_f1, val_pre = val_pre, val_recall = val_recall, acc = acc, f1 = f1, pre = pre, recall = recall, AUC=auc, AP=ap, GNN='FPGN')
    print('train_time:{:.4f}, val_time:{:.4f}, test_time:{:.4f}'.format(train_time, val_time, test_time))
    if(best_acc < val_acc):
        best_acc = val_acc
        best_f1, best_pre, best_recall = f1, pre, recall
        best_ap = ap
        best_epoch = epoch
        best_auc = auc
    if(args.tensorboard):
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_scalar('th', best_th, global_step=epoch)
        writer.add_scalar('auc', auc, global_step=epoch)
        writer.add_scalar('ap', ap, global_step=epoch)
        writer.add_scalar('acc', acc, global_step=epoch)
        writer.add_scalar('f1', f1, global_step=epoch)
        writer.add_scalar('pre', pre, global_step=epoch)
        writer.add_scalar('recall', recall, global_step=epoch)
        writer.add_scalar('best_auc', best_auc, global_step=epoch)
        writer.add_scalar('best_ap', best_ap, global_step=epoch)
        writer.add_scalar('best_acc', best_acc, global_step=epoch)
        writer.add_scalar('best_f1', best_f1, global_step=epoch)
        writer.add_scalar('best_pre', best_pre, global_step=epoch)
        writer.add_scalar('best_recall', best_recall, global_step=epoch)

writer.close()
print('th', best_th, 'best_acc', best_acc, 'best_f1', best_f1, 'best_pre', best_pre, 'best_recall', best_recall, 'best ap', best_ap, 'best_auc', best_auc, 'best epoch', best_epoch)
print('------------------------')