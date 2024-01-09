import torch 
import argparse
from utils import dotdict
from activation_dataset import setup_token_data
import wandb
import json
from datetime import datetime
from tqdm import tqdm
from einops import rearrange
# from standard_metrics import run_with_model_intervention, perplexity_under_reconstruction, mean_nonzero_activations
# Create 
# # make an argument parser directly below
# parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped")
# parser.add_argument("--layer", type=int, default=4)
# parser.add_argument("--setting", type=str, default="residual")
# parser.add_argument("--l1_alpha", type=float, default=3e-3)
# parser.add_argument("--num_epochs", type=int, default=10)
# parser.add_argument("--model_batch_size", type=int, default=4)
# parser.add_argument("--lr", type=float, default=1e-3)
# parser.add_argument("--kl", type=bool, default=False)
# parser.add_argument("--reconstruction", type=bool, default=False)
# parser.add_argument("--dataset_name", type=str, default="NeelNanda/pile-10k")
# parser.add_argument("--device", type=str, default="cuda:4")

# args = parser.parse_args()
cfg = dotdict()
# cfg.model_name="EleutherAI/pythia-70m-deduped"
# cfg.model_name="usvsnsp/pythia-6.9b-rm-full-hh-rlhf"
cfg.model_name="reciprocate/dahoas-gptj-rm-static"
# cfg.layers=[i for i in range(28)]
layers = 8
cfg.layers=[i for i in range(20,28)]
cfg.setting="residual"
# cfg.tensor_name="gpt_neox.layers.{layer}"
cfg.tensor_name="transformer.h.{layer}"
# linearly interpolate between 8e-4 & 1e-4
# cfg.l1_alpha=torch.linspace(1e-4, 1e-3, 8).tolist()
cfg.l1_alpha=torch.linspace(3.5e-3, 7e-3, 8).tolist()
cfg.sparsity=None
cfg.model_batch_size=4
cfg.lr=1e-3
cfg.kl=False
cfg.reconstruction=False
# cfg.dataset_name="NeelNanda/pile-10k"
# cfg.dataset_name="Skylion007/openwebtext"
cfg.dataset_name="Elriggs/openwebtext-100k"
cfg.device="cuda:0"
cfg.ratio = 4
# cfg.device="cpu"
tensor_names = [cfg.tensor_name.format(layer=layer) for layer in cfg.layers]
# Load in the model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

from datasets import load_dataset, concatenate_datasets
from activation_dataset import chunk_and_tokenize

# Load the datasets
dataset1 = load_dataset("Anthropic/hh-rlhf", split="train")
dataset2 = load_dataset("Elriggs/openwebtext-100k", split="train")

# Rename 'chosen' column in dataset1 to 'text' to match dataset2
dataset1 = dataset1.rename_column("chosen", "text")

# Concatenate the datasets
dataset = concatenate_datasets([dataset1, dataset2])

# Chunk & tokenize the dataset
max_seq_length= 256
dataset, _ = chunk_and_tokenize(dataset, tokenizer, max_length=max_seq_length)
max_tokens = dataset.num_rows*max_seq_length
print(f"Number of tokens: {max_tokens/1e6:.2f}M")

from torch.utils.data import DataLoader
batch_size = 16
token_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Run 1 datapoint on model to get the activation size (cause don't want to deal w/ different naming schemes in config files)
from baukit import Trace, TraceDict

text = "1"
tokens = tokenizer(text, return_tensors="pt").input_ids.to(cfg.device)
# Your activation name will be different. In the next cells, we will show you how to find it.
with torch.no_grad():
    with Trace(model, tensor_names[0]) as ret:
        _ = model(tokens)
        representation = ret.output
        # check if instance tuple
        if(isinstance(representation, tuple)):
            representation = representation[0]
        activation_size = representation.shape[-1]
print(f"Activation size: {activation_size}")

from torch import nn
from torchtyping import TensorType


class TiedSAE(nn.Module):
    def __init__(self, activation_size, n_dict_components):
        super().__init__()
        self.encoder = nn.Parameter(torch.empty((n_dict_components, activation_size)))
        nn.init.xavier_uniform_(self.encoder)
        self.encoder_bias = nn.Parameter(torch.zeros((n_dict_components,)))

    def get_learned_dict(self):
        norms = torch.norm(self.encoder, 2, dim=-1)
        return self.encoder / torch.clamp(norms, 1e-8)[:, None]

    def encode(self, batch):
        c = torch.einsum("nd,bd->bn", self.encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c

    def decode(self, code: TensorType["_batch_size", "_n_dict_components"]) -> TensorType["_batch_size", "_activation_size"]:
        learned_dict = self.get_learned_dict()
        x_hat = torch.einsum("nd,bn->bd", learned_dict, code)
        return x_hat

    def forward(self, batch: TensorType["_batch_size", "_activation_size"]) -> TensorType["_batch_size", "_activation_size"]:
        c = self.encode(batch)
        x_hat = self.decode(c)
        return x_hat, c

    def n_dict_components(self):
        return self.get_learned_dict().shape[0]

n_dict_components = activation_size*cfg.ratio
all_autoencoders = [TiedSAE(activation_size, n_dict_components) for _ in range(len(tensor_names))]

optimizers = [torch.optim.Adam(autoencoder.parameters(), lr=cfg.lr) for autoencoder in all_autoencoders]

print("WARNING: Only works on tied SAE")
# Wandb setup
secrets = json.load(open("secrets.json"))
wandb.login(key=secrets["wandb_key"])
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb_run_name = f"{cfg.model_name}_{start_time[4:]}_{cfg.sparsity}"  # trim year
print(f"wandb_run_name: {wandb_run_name}")
wandb.init(project="sparse coding", config=dict(cfg), name=wandb_run_name, entity="sparse_coding")

import numpy as np
# Make directory trained_models if it doesn't exist
import os
if not os.path.exists("trained_models"):
    os.makedirs("trained_models")
model_save_name = cfg.model_name.split("/")[-1]

num_batch = len(token_loader)
log_space = np.logspace(0, np.log10(num_batch), 11)  # 11 to get 10 intervals
save_batches = [int(x) for x in log_space[1:]]  # Skip the first (0th) interval

dead_features = [torch.zeros(n_dict_components) for _ in range(len(tensor_names))]
last_encoders = [autoencoder.encoder.clone().detach() for autoencoder in all_autoencoders]
# max_num_tokens = 100000000
# Freeze model parameters 
model.eval()
model.requires_grad_(False)
for i, batch in enumerate(tqdm(token_loader)):
    if i > 5: # for testing code works
        break
    tokens = batch["input_ids"].to(cfg.device)
    with torch.no_grad(): # As long as not doing KL divergence, don't need gradients for model
        with TraceDict(model, tensor_names) as ret:
            _ = model(tokens)

    for auto_ind in range(len(tensor_names)):
        # Index into correct autoencoder, optimizer, and tensor_name
        autoencoder = all_autoencoders[auto_ind].to(cfg.device)
        optimizer = optimizers[auto_ind]
        tensor_name = tensor_names[auto_ind]
        dead_feature = dead_features[auto_ind]
        last_encoder = last_encoders[auto_ind]
        l1_alpha = cfg.l1_alpha[auto_ind]

        # Get intermediate layer activations
        representation = ret[tensor_name].output
        if(isinstance(representation, tuple)):
            representation = representation[0]
        layer_activations = rearrange(representation, "b seq d_model -> (b seq) d_model")
        
        # Run through autoencoder
        c = autoencoder.encode(layer_activations)
        x_hat = autoencoder.decode(c)

        # Calculate loss
        reconstruction_loss = (x_hat - layer_activations).pow(2).mean()
        l1_loss = torch.norm(c, 1, dim=-1).mean()
        total_loss = reconstruction_loss + l1_alpha*l1_loss

        # Update dead features
        dead_feature += c.sum(dim=0).cpu()

        
        # Log
        if (i % 50 == 0): # Check here so first check is model w/o change
            # self_similarity = torch.cosine_similarity(c, last_encoder, dim=-1).mean().cpu().item()
            # Above is wrong, should be similarity between encoder and last encoder
        
            num_tokens_so_far = i*max_seq_length*batch_size
            with torch.no_grad():
                # print the norm of layer_activations
                # print(f"Layer {auto_ind} | Layer Activation Norm: {torch.norm(layer_activations, 2, dim=-1).mean().cpu().item()}")
                self_similarity = torch.cosine_similarity(autoencoder.encoder, last_encoder.to(cfg.device), dim=-1).mean().cpu().item()
                last_encoders[auto_ind] = autoencoder.encoder.clone().to("cpu")
                sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
                # Count number of dead_features are zero
                num_dead_features = (dead_feature == 0).sum().item()
            print(f"Layer {auto_ind} | Sparsity: {sparsity:.1f} | Dead Features: {num_dead_features} | Total Loss: {total_loss:.2f} | Reconstruction Loss: {reconstruction_loss:.2f} | L1 Loss: {l1_alpha*l1_loss:.2f} | l1_alpha: {l1_alpha:.2e} | Tokens: {num_tokens_so_far} | Self Similarity: {self_similarity:.2f}")
            wandb.log({f"Layer {auto_ind}": {
                'Sparsity': sparsity,
                'Dead Features': num_dead_features,
                'Total Loss': total_loss.item(),
                'Reconstruction Loss': reconstruction_loss.item(),
                'L1 Loss': (l1_alpha*l1_loss).item(),
                'l1_alpha': l1_alpha,
                'Tokens': num_tokens_so_far,
                'Self Similarity': self_similarity,
                'step': i}
            })
            dead_feature = torch.zeros(autoencoder.encoder.shape[0])

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Put SAE back on cpu
        all_autoencoders[auto_ind] = autoencoder.to("cpu")

wandb.finish()

# Save every autoencoder
for i, autoencoder in enumerate(all_autoencoders):
    save_name = f"{model_save_name}_r{cfg.ratio}_{tensor_names[i]}"  # trim year
    torch.save(autoencoder, f"trained_models/{save_name}.pt")