import os.path as osp
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
import random
from torch.nn import Embedding
import pyabcore
import math
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def attr_norm(array):
    if(array.max() - array.min() == 0):
        return np.zeros_like(array)
    return (array - array.min()) / (array.max() - array.min())

def count_gt(tf_dict, num_dict, time_dict, u, nu, d_time):
    if(u not in tf_dict):
        tf_dict[u] = set()
        num_dict[u] = dict()
        time_dict[u] = dict()
    if(nu not in num_dict[u]):
        num_dict[u][nu] = 0
    if(nu not in time_dict[u]):
        time_dict[u][nu] = list()

    tf_dict[u].update([nu])
    num_dict[u][nu] += 1
    time_dict[u][nu].append(d_time)

    return tf_dict, num_dict, time_dict

def get_gt(data, time_window, train_set = False, fal_flag = False):
    
    time_window = time_window.detach().numpy()
    G = nx.MultiGraph()
    G.add_weighted_edges_from(data, weight='timestamp')

    u_set = set(data[:,0])


    truth = dict()
    false_before = dict()
    false_after = dict() 
    truth_num = dict() 
    false_before_num = dict() 
    false_after_num = dict()
    pos_delta_time = dict() 
    false_before_delta_time = dict() 
    false_after_delta_time = dict() 
    print('checking u...')

    for u in list(u_set):
        for p, t in G[u].items():
            t_list = []
            for i in t:
                t_list.append(t[i]['timestamp'])
            for nu, nt in G[p].items():
                if(nu == u):
                    continue
                nt_list = []
                for i in nt:
                    nt_list.append(nt[i]['timestamp'])
                d_times = np.subtract.outer(t_list, nt_list).flatten()
                min_d_time = d_times[np.argmin(np.abs(d_times))]
                in_time_times = d_times[ (0 <= d_times) * (d_times <= time_window)]

                if(len(in_time_times) > 0): 
                    d_time = in_time_times.min()
                    truth, truth_num, pos_delta_time = count_gt(truth, truth_num, pos_delta_time, u, nu, d_time) 
                elif(min_d_time < 0):
                    false_before, false_before_num, false_before_delta_time = count_gt(false_before, false_before_num, false_before_delta_time, u, nu, min_d_time)
                elif(min_d_time > time_window):
                    false_after, false_after_num, false_after_delta_time = count_gt(false_after, false_after_num, false_after_delta_time, u, nu, min_d_time)
    
    all_pos_ground_truth = list()
    all_neg_ground_truth = list()

    pos_edge_attr = list()
    pos_edge_time = list()


    print('get pos and neg..')
    for u in truth:
        pos_ground_truth = list()
        neg_ground_truth = list()

        pos = list(truth[u])
        pos_len = len(pos)
        neg_len = pos_len * 2

        neg_set = u_set - truth[u]

        assert len(neg_set) >= neg_len
        if(len(neg_set) == neg_len):
            neg = neg_set
        else:
            pn_total_set = set()
            if(u in false_before):
                pn_total_set.update(false_before[u])
            if(u in false_after):
                pn_total_set.update(false_after[u])
            pn_total_set = pn_total_set - truth[u] 

            out_window_neg = neg_set - pn_total_set 
            in_window_neg_len = len(pn_total_set)
            out_window_neg_len = len(out_window_neg)
            if(in_window_neg_len == pos_len and out_window_neg_len == pos_len):
                neg = neg_set
            elif(in_window_neg_len > pos_len and out_window_neg_len > pos_len):
                neg = set(random.sample(list(pn_total_set), pos_len)) | set(random.sample(list(out_window_neg), pos_len))
            elif(in_window_neg_len <= pos_len):
                neg = pn_total_set | set(random.sample(list(out_window_neg), neg_len - in_window_neg_len))
            elif(out_window_neg_len <= pos_len):
                neg = out_window_neg | set(random.sample(list(pn_total_set), neg_len - out_window_neg_len))
            assert len(neg) == neg_len
        for n in neg:
            neg_ground_truth.append([u, n])

        assert len(neg_ground_truth) == neg_len

        for pn in pos:
            pos_ground_truth.append([u, pn])
            if(train_set):
                # 训练集提供特征
                pos_edge_attr.append(truth_num[u][pn])
                pos_edge_time.append([np.mean(pos_delta_time[u][pn]), np.var(pos_delta_time[u][pn])])

        all_pos_ground_truth += pos_ground_truth
        all_neg_ground_truth += neg_ground_truth

    if(train_set):
        pos_edge_attr = attr_norm(np.array(pos_edge_attr))
        pos_edge_time = np.array(pos_edge_time)
        mean_pos_edge_time, var_pos_edge_time = pos_edge_time[:,0], pos_edge_time[:,1]
        mean_pos_edge_time = (mean_pos_edge_time - mean_pos_edge_time.min()) / (mean_pos_edge_time.max() - mean_pos_edge_time.min())
        var_pos_edge_time = (var_pos_edge_time - var_pos_edge_time.min()) / (var_pos_edge_time.max() - var_pos_edge_time.min())
        pos_edge_attr = np.column_stack((pos_edge_attr, mean_pos_edge_time, var_pos_edge_time)) 
        return torch.LongTensor(all_pos_ground_truth), torch.LongTensor(all_neg_ground_truth), torch.FloatTensor(pos_edge_attr)
    else:
        return torch.LongTensor(all_pos_ground_truth), torch.LongTensor(all_neg_ground_truth)


def main(data_name, test=None):
    print('-----------------')
    if(test != None):
        print('test_' + data_name)
    else:
        print(data_name)
    print('read data...')
    df = pd.read_csv(root_path+'/../original_data/'+data_name + '.csv', usecols=[0,1,2], names=['u', 'p', 't'], sep=',', dtype={'u':int, 'p':int, 't':float},header=0, encoding="utf-8", nrows=test, comment='%')
    df.drop_duplicates(keep='first',inplace=True)
    print('data len', len(df))
    print('check node...')
    
    df_e = torch.LongTensor(np.array(df[['u', 'p']], dtype=int))
    data_len = df_e.shape[0]
    df_e[:,1] += (df_e[:,0].max()+1)
    node_list = df_e.unique()
    node_list, _ = node_list.sort()
    node_num = len(node_list)

    assoc = torch.empty(node_list.max()+1, dtype=torch.long)
    assoc[node_list] = torch.arange(node_num)
    df_e = assoc[df_e]

    df_t = torch.FloatTensor(np.array(df['t'], dtype=float))
    max_time, min_time = df_t.max(), df_t.min()
    time_window = (max_time - min_time) * time_window_rate
    df_t = df_t - min_time
    df_t, sort_idx = df_t.sort()
    df_e = df_e[sort_idx]

    df = torch.cat((df_e, df_t.to(int).unsqueeze(-1)),1)

    train_data_array = df[:int(data_len * train_len)]

    train_u_set = train_data_array[:,0].unique()
    train_data_array = torch.from_numpy(np.array(train_data_array))

    print('remove test edge if its u can not be found in train...')
    tmp_test_data_array = df[int(data_len * train_len):]
    test_delete_num = 0
    test_data_array = []
    for tmp_u, tmp_i, tmp_t in tmp_test_data_array:
        if(tmp_u not in train_u_set):
            test_delete_num += 1
            continue
        test_data_array.append([tmp_u, tmp_i, tmp_t])
    print('test_delete_num:', test_delete_num, 'remain test:', len(test_data_array))
    test_data_array = torch.from_numpy(np.array(test_data_array))

    train_u_set = train_data_array[:,0].unique()
    train_u_set, _ = train_u_set.sort()
    test_u_set = test_data_array[:,0].unique()
    assert len(train_u_set) >= len(test_u_set) and train_u_set.max() >= test_u_set.max()

    assoc = torch.empty(train_u_set.max()+1, dtype=torch.long)
    assoc[train_u_set] = torch.arange(len(train_u_set))
    train_data_array[:,0] = assoc[train_data_array[:,0]]
    test_data_array[:,0] = assoc[test_data_array[:,0]]
    train_u_set = assoc[train_u_set]

    p_set = train_data_array[:,1].unique()
    p_set, _ = p_set.sort()
    assoc = torch.empty(p_set.max()+1, dtype=torch.long)
    assoc[p_set] = torch.arange(len(p_set)) + train_data_array[:,0].max() + 1
    train_data_array[:,1] = assoc[train_data_array[:,1]]

    assert len(train_data_array[:,0].unique()) == train_data_array[:,0].unique().max()+1
    src_max = int(train_data_array[:,0].max()+1)

    dst_max = train_data_array[:,1].unique()
    assert len(dst_max) == dst_max.max() - (src_max-1) 
    
    dst_max = int(dst_max.max()+1)

    node_list = train_data_array[:,:2].unique()
    node_num = len(node_list)

    train_data_array = np.array(train_data_array)
    test_data_array = np.array(test_data_array)

    print('src num', src_max, 'dst num', dst_max - (src_max-1))

    print('get edge/node features...')
    abcore = pyabcore.Pyabcore(src_max, dst_max)
    abcore.index(train_data_array[:,:2].astype(np.int32))
    core_u_x = torch.BoolTensor([])
    core_p_x = torch.BoolTensor([])

    a = 2
    for b in range(1, 21):
        abcore.query(a, b)
        result_u = torch.BoolTensor(abcore.get_left())
        result_p = torch.BoolTensor(abcore.get_right())
        core_u_x = torch.cat((core_u_x, result_u.unsqueeze(-1)),dim=1)
        core_p_x = torch.cat((core_p_x, result_p.unsqueeze(-1)),dim=1)

    embedding = Embedding(max(src_max, dst_max), 32)
    node_feature = embedding(torch.arange(dst_max).to(int))
    node_feature = torch.cat((torch.zeros(len(node_feature), 1), node_feature),1)
    node_feature[:len(core_u_x)][:,0] = 1.
    core_x = core_p_x
    core_x[:len(core_u_x)] = core_u_x
    node_feature_with_core = torch.cat((node_feature, core_x), 1)
    assert len(node_feature) == len(node_feature_with_core) == len(node_list)

    msg = torch.cat((node_feature[train_data_array[:,0]], node_feature[train_data_array[:,1]]),1)
    msg_core = torch.cat((node_feature_with_core[train_data_array[:,0]], node_feature_with_core[train_data_array[:,1]]),1)
    assert len(msg) == len(train_data_array)
    
    print('get pos and neg')
    data = Data()
    data.num_nodes = node_num
    data.pos_data, data.neg_data, data.pos_attr = get_gt(train_data_array, time_window, train_set=True)
    data.test_pos_data, data.test_neg_data = get_gt(test_data_array, time_window)

    print('pos_edge', len(data.pos_data), 'neg_edge', len(data.neg_data))
    print('test_gt_pos', len(data.test_pos_data), 'test_gt_neg', len(data.test_neg_data))

    train_data_array = torch.tensor(train_data_array)

    data.abcore_feature_dim = core_p_x.shape[1]
    data.x = node_feature 
    data.x_core = node_feature_with_core
    data.edge_index = train_data_array[:,:2].T
    data.src = train_data_array[:,0]
    data.dst = train_data_array[:,1]
    data.t = train_data_array[:,2]
    data.msg = msg
    data.msg_core = msg_core



    train_val_len = len(data.pos_data)
    tmp_train_len = math.ceil(train_val_len * 0.8)
    print('pos_train_len', tmp_train_len, 'neg_train_len', tmp_train_len)
    print('pos_val_len', train_val_len - tmp_train_len, 'neg_val_len', train_val_len - tmp_train_len)
    train_index = torch.LongTensor(random.sample(range(train_val_len), tmp_train_len))
    train_mask = torch.zeros(train_val_len)
    train_mask[train_index] = 1
    val_mask = 1 - train_mask
    train_mask = train_mask.to(bool)
    val_mask = val_mask.to(bool)

    neg_train_val_len = len(data.neg_data)
    neg_train_index = torch.LongTensor(random.sample(range(neg_train_val_len), tmp_train_len * 2))
    neg_train_mask = torch.zeros(neg_train_val_len)
    neg_train_mask[neg_train_index] = 1
    neg_val_mask = 1 - neg_train_mask
    neg_train_mask = neg_train_mask.to(bool)
    neg_val_mask = neg_val_mask.to(bool)

    data.train_pos_data = data.pos_data[train_mask]
    data.val_pos_data = data.pos_data[val_mask]

    data.train_pos_attr = data.pos_attr[train_mask]
    data.val_pos_attr = data.pos_attr[val_mask]


    data.train_neg_data = data.neg_data[neg_train_mask]
    data.val_neg_data = data.neg_data[neg_val_mask]

    data.train_data = torch.cat((data.train_pos_data, data.train_neg_data))
    data.val_data = torch.cat((data.val_pos_data, data.val_neg_data))
    data.test_data = torch.cat((data.test_pos_data, data.test_neg_data))

    data.train_y = torch.cat((torch.ones(len(data.train_pos_data)), torch.zeros(len(data.train_neg_data)))).to(int)
    data.val_y = torch.cat((torch.ones(len(data.val_pos_data)), torch.zeros(len(data.val_neg_data)))).to(int)
    data.test_y = torch.cat((torch.ones(len(data.test_pos_data)), torch.zeros(len(data.test_neg_data)))).to(int)

    assert len(data.train_pos_data) * 2 == len(data.train_neg_data)
    assert len(data.val_pos_data) * 2 == len(data.val_neg_data)
    assert len(data.test_pos_data) * 2 == len(data.test_neg_data)

    if(test != None):
        torch.save(data, root_path+'/test_'+data_name+'.pt')
    else:
        torch.save(data, root_path+'/'+data_name+'.pt')

    print('finished')
    print('-----------------')

if __name__ == '__main__':
    setup_seed(10)
    root_path = osp.dirname(osp.realpath(__file__))
    train_len = 0.8
    
    data_name = ['wikipedia', 'reddit']
    time_window_rates = [0.0001, 0.0001]
    for i in range(len(data_name)):
        time_window_rate = time_window_rates[i]
        print('time_window_rate', time_window_rate)
        main(data_name[i])
    