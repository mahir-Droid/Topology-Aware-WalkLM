from transformers import AutoTokenizer, AutoModel, T5Model, AutoModelWithLMHead
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForPreTraining, BertForPreTraining, BertConfig
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import torch
import random
import os
import math
import pickle as pickle
import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0")

# Load node data
op1, op2, op3 = [], [], []
with open('./PubMed/node.dat', 'r') as original_meta_file:
    for line in original_meta_file:
        temp1, temp2, temp3 = line.split('\t')
        op1.append(int(temp1))
        op2.append(temp2)
        op3.append(temp3[:-1])

# Create graph structure
G = [[] for i in range(len(op3))]
with open('./PubMed/link.dat', 'r') as original_meta_file:
    for line in original_meta_file:
        start, end, edge_type, edge_class = line.split('\t')
        G[int(start)].append([int(end), int(edge_type)])

# Generate random walks
line_idx = op1
rand = random.Random()
patient_patient_path = []
alpha = 0.05
path_length = 1000000
path_num = 450000

dic = {}
for line in range(path_num):
    temp_path = []
    start_path = rand.choice(line_idx)
    temp_path.append([start_path, -1])
    dic[start_path] = 1
    for i in range(path_length):
        cur = temp_path[-1][0]
        if len(G[cur]) > 0:
            if rand.random() >= alpha:
                cur_path = rand.choice(G[cur])
                temp_path.append(cur_path)
                dic[cur_path[0]] = 1
            else:
                break
        else:
            break
    if len(temp_path) >= 2:
        patient_patient_path.append(temp_path)

# Define edge type names
line_name = {
    0: "and", 1: "causing", 2: "and", 3: "in", 4: "in",
    5: "and", 6: "in", 7: "with", 8: "with", 9: "and"
}

# Generate text sequences
with open('./PubMed/output.txt', 'w') as f:
    for i in range(len(patient_patient_path)):
        print(op2[patient_patient_path[i][0][0]], line_name[patient_patient_path[i][1][1]], 
              op2[patient_patient_path[i][1][0]], end='', file=f)
        for j in range(1, len(patient_patient_path[i])-2):
            print(' ' + line_name[patient_patient_path[i][j+1][1]], 
                  op2[patient_patient_path[i][j+1][0]], end='', file=f)
        if len(patient_patient_path[i]) > 2:    
            print(' ' + line_name[patient_patient_path[i][-1][1]], 
                  op2[patient_patient_path[i][-1][0]], end='', file=f)
        print("\n", end='', file=f)

# Load and split dataset
with open('./PubMed/output.txt', 'r') as file:
    corpus = [line.rstrip("\n") for line in file.readlines()]

train_text, val_text = train_test_split(corpus, test_size=0.15, random_state=42)

with open('./PubMed/train_corpus.txt', 'w') as file:
    for paragraph in train_text:
        file.write(paragraph + "\n")
        
with open('./PubMed/val_corpus.txt', 'w') as file:
    for paragraph in val_text:
        file.write(paragraph + "\n")

datasets = load_dataset("text", data_files={
    "train": './PubMed/train_corpus.txt',
    "validation": './PubMed/val_corpus.txt'
})

class TopologyAwareTrainer(Trainer):
    def __init__(self, *args, graph=None, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = graph
        self.alpha = alpha
        
    def compute_structural_similarity(self, node_ids):
        n = len(node_ids)
        sim_matrix = torch.zeros((n, n), device=self.args.device)
        
        for i, node in enumerate(node_ids):
            neighbors = set([edge[0] for edge in self.graph[node]])
            for j, other_node in enumerate(node_ids):
                if other_node in neighbors:
                    sim_matrix[i][j] = 1.0
                elif node == other_node:
                    sim_matrix[i][j] = 1.0
                else:
                    sim_matrix[i][j] = 0.1
        return sim_matrix

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        mlm_loss = outputs.loss
        
        hidden_states = outputs.hidden_states[-1]
        node_embeddings = hidden_states[:, 0, :]
        
        # Extract node IDs from input texts
        batch_texts = self.tokenizer.batch_decode(inputs['input_ids'])
        batch_node_ids = [int(text.split()[0]) for text in batch_texts]
        
        structural_sim = self.compute_structural_similarity(batch_node_ids)
        
        emb_sim = F.cosine_similarity(
            node_embeddings.unsqueeze(1),
            node_embeddings.unsqueeze(0),
            dim=-1
        )
        
        struct_loss = F.mse_loss(emb_sim, structural_sim)
        total_loss = (1 - self.alpha) * mlm_loss + self.alpha * struct_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

# Initialize model and tokenizer
card = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(card, use_fast=True)
model = AutoModelForPreTraining.from_pretrained(card)

def tokenize_function(samples):
    return tokenizer(samples["text"], truncation=True)

tokenized_datasets = datasets.map(
    tokenize_function, 
    batched=True, 
    num_proc=4, 
    remove_columns=["text"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="model/topology_aware",
    overwrite_output_dir=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    dataloader_num_workers=8,
    load_best_model_at_end=True,
    gradient_accumulation_steps=20,
    num_train_epochs=6,
    learning_rate=0.0005,
    weight_decay=0.01
)

trainer = TopologyAwareTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    graph=G,
    alpha=0.3
)

# Train and save model
trainer.train()
trainer.evaluate()
trainer.save_model("model/xyz")

# Generate embeddings
model_name = "model/xyz"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def get_word_embeddings(word, device):
    encoded_word = tokenizer.encode(word, add_special_tokens=False)
    tokens_tensor = torch.tensor([encoded_word]).to(device)
    with torch.no_grad():
        output = model(tokens_tensor)
        embeddings = output[0][0].mean(dim=0)
    return embeddings.cpu().numpy()

# Initialize embeddings
emb = get_word_embeddings("hello", device)
emb = np.zeros((len(op2), len(emb)))

# Generate embeddings for all nodes
for i in range(len(op2)):
    word = op2[i]
    emb[i] = get_word_embeddings(word, device)

# Save embeddings
with open('./PubMed/emb.dat', 'w') as file:
    file.write('pubmed\n')
    for i in range(len(op2)):
        file.write(f'{i}\t')
        file.write(' '.join(emb[i].astype(str)))
        file.write('\n')