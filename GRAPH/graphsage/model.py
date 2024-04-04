import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

import json

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
torch.autograd.set_detect_anomaly(True)

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())



class UnsupervisedGraphSage(nn.Module):
    def __init__(self, enc, adj_list,num_nodes,feat_data):
        super(UnsupervisedGraphSage, self).__init__()
        self.enc = enc
        self.adj_list = adj_list
        self.num_nodes = num_nodes
        self.all_nodes = set(range(self.num_nodes))
        self.num_samples = 5  # Number of negative samples
        self.feat_data = feat_data
        #self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        #init.xavier_uniform(self.weight)


    def forward(self, nodes):
        embeds = self.enc(nodes)
        return embeds.t()

    def loss(self, nodes):
        embeds = self.forward(nodes)
        loss = 0.0
        
        for node_idx, node in enumerate(nodes):


            node_embed = embeds[node_idx]
            neighbors = set(self.adj_list[node])
            
            # Positive samples loss
            for neighbor in neighbors:
                neighbor_embed = self.feat_data[neighbor]
                similarity = torch.sigmoid(torch.dot(node_embed.squeeze(), neighbor_embed.squeeze()))
                loss -= torch.log(similarity)
                #print(loss)
            
            # Negative samples loss
            non_neighbors = list(self.all_nodes - neighbors - {node})
            neg_samples = random.sample(non_neighbors, min(self.num_samples, len(non_neighbors)))
            for neg_sample in neg_samples:
                neg_sample_embed = self.feat_data[neg_sample]
                similarity = torch.sigmoid(torch.dot(node_embed.squeeze(), neg_sample_embed.squeeze()))
                # Subtract because we want to minimize the similarity for negative samples
                loss -= torch.log(1 - similarity)
        
        # Average the loss
        loss = loss / len(nodes)
        return loss

FILE_PATH = "paperdata/papers_test.json"


def get_data():
    with open(FILE_PATH) as f:
        for line in f:
            yield line


def load_paper_data():
    num_nodes = 630000
    num_feats = 128
    feat_data = np.zeros((num_nodes, num_feats))
    # labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}

    mean = 0
    std_dev = 1
    size = (1, 128)

    year_limit = 2000

    adj_lists = defaultdict(set)
    data = get_data()
    for i, paper in enumerate(data):
        paper = json.loads(paper)
        try:
            feat_data[i, :] = np.random.normal(mean, std_dev, size)
            paper1 = paper["id"]
            referVec = paper["citations"]
            adj_lists[paper1] = referVec
        except:
            pass

    adj_file_path = "adj_list.json"

    # 将字典保存为 JSON 文件
    with open(adj_file_path, "w") as json_file:
        json.dump(adj_lists, json_file)

    print("JSON file saved:", adj_file_path)

    return feat_data, adj_lists


def run_unsupervised():
    np.random.seed(1)
    random.seed(1)
    feat_data, adj_lists = load_paper_data()
    feat_data = torch.FloatTensor(feat_data)
    num_nodes = len(adj_lists.keys())
    features = nn.Embedding(num_nodes, 128)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 128, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(
        lambda nodes: enc1(nodes).t(),
        enc1.embed_dim,
        128,
        adj_lists,
        agg2,
        base_model=enc1,
        gcn=True,
        cuda=False,
    )
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = UnsupervisedGraphSage(enc2,adj_lists,num_nodes,feat_data)
    #    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    train = list(rand_indices[1500:])

    train = []
    non_empty_key = 0

    for key in adj_lists.keys():
        if len(adj_lists[key])!=0:
            train.append(key)
            non_empty_key += 1
    
    print(non_empty_key)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7
    )
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(
            batch_nodes
        )
        updated_feat = graphsage(batch_nodes)

        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(batch, loss.data.item())
        
        if batch % 10 == 0:  # Check if the current batch is a multiple of 10
            print(f"Batch {batch}, Loss: {loss.data.item()}") 

    print("Average batch time:", np.mean(times))

    return


if __name__ == "__main__":
    run_unsupervised()