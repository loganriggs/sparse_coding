#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch 
import argparse
from utils import dotdict
from activation_dataset import setup_token_data
import wandb
import json
from datetime import datetime
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt

cfg = dotdict()
# models: "EleutherAI/pythia-6.9b", "lomahony/eleuther-pythia6.9b-hh-sft", "usvsnsp/pythia-6.9b-ppo", "Dahoas/gptj-rm-static", "reciprocate/dahoas-gptj-rm-static"
# cfg.model_name="lomahony/eleuther-pythia6.9b-hh-sft"
# "EleutherAI/pythia-70m", "lomahony/pythia-70m-helpful-sft", "lomahony/eleuther-pythia70m-hh-sft"
cfg.model_name="EleutherAI/pythia-70m"
cfg.layers=[0,1,2,3,4,5]
cfg.setting="residual"
# cfg.tensor_name="gpt_neox.layers.{layer}" or "transformer.h.{layer}"
cfg.tensor_name="gpt_neox.layers.{layer}"
original_l1_alpha = 8e-4
# cfg.l1_alpha=original_l1_alpha
# cfg.l1_alphas=[0, 1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1e-3, 2e-3, 4e-3, 8e-3]
cfg.l1_alphas=[2e-3,]
cfg.sparsity=None
cfg.num_epochs=10
cfg.model_batch_size=8 * 8
cfg.lr=1e-3
cfg.kl=False
cfg.reconstruction=False
# cfg.dataset_name="NeelNanda/pile-10k"
cfg.dataset_name="Elriggs/openwebtext-100k"
cfg.device="cuda:0"
cfg.ratio = 4
cfg.seed = 0
# cfg.device="cpu"


# In[ ]:


tensor_names = [cfg.tensor_name.format(layer=layer) for layer in cfg.layers]


# In[ ]:


# Load in the model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, GPTJForSequenceClassification
model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
model = model.to(cfg.device)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)


# In[ ]:


# Download the dataset
# TODO iteratively grab dataset?
cfg.max_length = 256
token_loader = setup_token_data(cfg, tokenizer, model, seed=cfg.seed)
num_tokens = cfg.max_length*cfg.model_batch_size*len(token_loader)
print(f"Number of tokens: {num_tokens}")


# In[ ]:


# Run 1 datapoint on model to get the activation size
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


# In[ ]:


# Initialize New autoencoder
from autoencoders.learned_dict import TiedSAE, UntiedSAE, AnthropicSAE
from torch import nn
autoencoders = []
optimizers = []
for layer in cfg.layers:
    l1_variants = []
    l1_optimizers = []
    for l1 in cfg.l1_alphas:
        params = dict()
        n_dict_components = activation_size*cfg.ratio
        params["encoder"] = torch.empty((n_dict_components, activation_size), device=cfg.device)
        nn.init.xavier_uniform_(params["encoder"])

        params["decoder"] = torch.empty((n_dict_components, activation_size), device=cfg.device)
        nn.init.xavier_uniform_(params["decoder"])

        params["encoder_bias"] = torch.empty((n_dict_components,), device=cfg.device)
        nn.init.zeros_(params["encoder_bias"])

        params["shift_bias"] = torch.empty((activation_size,), device=cfg.device)
        nn.init.zeros_(params["shift_bias"])

        autoencoder = AnthropicSAE(  # TiedSAE, UntiedSAE, AnthropicSAE
            # n_feats = n_dict_components, 
            # activation_size=activation_size,
            encoder=params["encoder"],
            encoder_bias=params["encoder_bias"],
            decoder=params["decoder"],
            shift_bias=params["shift_bias"],
        )
        
        # autoencoder = torch.load(f"/root/sparse_coding/trained_models/base_sae_70m/base_sae_70m_{layer}_{l1}.pt")
        
        autoencoder.to_device(cfg.device)
        autoencoder.set_grad()
        l1_variants.append(autoencoder)

        optimizer = torch.optim.Adam(
            [
                autoencoder.encoder, 
                autoencoder.encoder_bias,
                autoencoder.decoder,
                autoencoder.shift_bias,
            ], lr=cfg.lr)
        l1_optimizers.append(optimizer)
        
    autoencoders.append(l1_variants)
    optimizers.append(l1_optimizers)


# In[ ]:


# Set target sparsity to 10% of activation_size if not set
if cfg.sparsity is None:
    cfg.sparsity = int(activation_size*0.05)
    print(f"Target sparsity: {cfg.sparsity}")

target_lower_sparsity = cfg.sparsity * 0.9
target_upper_sparsity = cfg.sparsity * 1.1
adjustment_factor = 0.1  # You can set this to whatever you like


# In[ ]:


original_bias = autoencoder.encoder_bias.clone().detach()
# Wandb setup
secrets = json.load(open("secrets.json"))
wandb.login(key=secrets["wandb_key"])
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb_run_name = f"{cfg.model_name}_{start_time[4:]}_{cfg.sparsity}"  # trim year
print(f"wandb_run_name: {wandb_run_name}")


# In[ ]:


wandb.init(project="sparse coding", config=dict(cfg), name=wandb_run_name)


# In[ ]:


# time_since_activation = torch.zeros(autoencoder.encoder.shape[0])
# total_activations = torch.zeros(autoencoder.encoder.shape[0])
max_num_tokens = 100_000_000
save_every = 30_000
num_saved_so_far = 0
# Freeze model parameters 
model.eval()
model.requires_grad_(False)
model.to(cfg.device)
# last_encoder = autoencoder.encoder.clone().detach()
i=0
for epoch in range(cfg.num_epochs):
    for i_inside, batch in enumerate(tqdm(token_loader,total=max(int(max_num_tokens/(cfg.max_length*cfg.model_batch_size)), len(token_loader)))):
        i+=1
        tokens = batch["input_ids"].to(cfg.device)
        with torch.no_grad(): # As long as not doing KL divergence, don't need gradients for model
                    
            layer_activations = []
            with TraceDict(model, tensor_names) as ret:
                _ = model(tokens)
                for layer_name in tensor_names:
                    representation = ret[layer_name].output[0]
                    flattened_rep = rearrange(representation, "b seq d_model -> (b seq) d_model")
                    layer_activations.append(flattened_rep)
                    
        # activation_saver.save_batch(layer_activations.clone().cpu().detach())
        
        wandb_log = {}
        
        for layer in range(len(cfg.layers)):
            for l1 in range(len(cfg.l1_alphas)):
                l1_alpha = cfg.l1_alphas[l1]
                acts = layer_activations[layer]
                autoencoder = autoencoders[layer][l1]
                optimizer = optimizers[layer][l1]
                
                c = autoencoder.encode(acts)
                x_hat = autoencoder.decode(c)
                
                reconstruction_loss = (x_hat - acts).pow(2).mean()
                l1_loss = torch.norm(c, 1, dim=-1).mean()
                total_loss = reconstruction_loss + l1_alpha*l1_loss

                # time_since_activation += 1
                # time_since_activation = time_since_activation * (c.sum(dim=0).cpu()==0)
                # total_activations += c.sum(dim=0).cpu()
                if ((i) % 10 == 0): # Check here so first check is model w/o change
                    # self_similarity = torch.cosine_similarity(c, last_encoder, dim=-1).mean().cpu().item()
                    # Above is wrong, should be similarity between encoder and last encoder
                    # self_similarity = torch.cosine_similarity(autoencoder.encoder, last_encoder, dim=-1).mean().cpu().item()
                    # last_encoder = autoencoder.encoder.clone().detach()
                    num_tokens_so_far = i*cfg.max_length*cfg.model_batch_size
                    with torch.no_grad():
                        sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
                        # Count number of dead_features are zero
                        # num_dead_features = (time_since_activation >= min(i, 200)).sum().item()
                    # print(f"Sparsity: {sparsity:.1f} | Dead Features: {num_dead_features} | Total Loss: {total_loss:.2f} | Reconstruction Loss: {reconstruction_loss:.2f} | L1 Loss: {cfg.l1_alpha*l1_loss:.2f} | l1_alpha: {cfg.l1_alpha:.2e} | Tokens: {num_tokens_so_far} | Self Similarity: {self_similarity:.2f}")
                    wandb_log.update({
                        f'Sparsity {layer}, {l1_alpha}': sparsity,
                        # 'Dead Features': num_dead_features,
                        f'Total Loss {layer}, {l1_alpha}': total_loss.item(),
                        f'Reconstruction Loss {layer}, {l1_alpha}': reconstruction_loss.item(),
                        f'L1 Loss {layer}, {l1_alpha}': (l1_alpha*l1_loss).item(),
                        f'Raw L1 {layer}, {l1_alpha}': (l1_loss).item(),
                        f'Tokens': num_tokens_so_far,
                        # 'Self Similarity': self_similarity
                    })
                    
                    # dead_features = torch.zeros(autoencoder.encoder.shape[0])
                    
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                    
        if len(wandb_log) > 0:
            wandb.log(wandb_log)

        inside_tokens = i_inside*cfg.max_length*cfg.model_batch_size
        if inside_tokens > max_num_tokens:
            break
        
    # SAVE SAEs each epoch
    print(f"SAVED SAEs for EPOCH {epoch}")
    import os
    if not os.path.exists(f"trained_models/checkpoint{epoch}"):
        os.makedirs(f"trained_models/checkpoint{epoch}")
    # Save model

    for layer in range(len(cfg.layers)):
        for l1 in range(len(cfg.l1_alphas)):
            l1_alpha = cfg.l1_alphas[l1]
            
            autoencoder = autoencoders[layer][l1]
            model_save_name = cfg.model_name.split("/")[-1]
            save_name = f"base_sae_70m_{layer}_{l1_alpha}"
            torch.save(autoencoder, f"trained_models/checkpoint{epoch}/{save_name}.pt")


# In[ ]:


# model_save_name = cfg.model_name.split("/")[-1]
# save_name = f"{model_save_name}_sp{cfg.sparsity}_r{cfg.ratio}_{tensor_names[0]}"

# Make directory traiend_models if it doesn't exist
import os
if not os.path.exists("trained_models"):
    os.makedirs("trained_models")
# Save model

for layer in range(len(cfg.layers)):
    for l1 in range(len(cfg.l1_alphas)):
        l1_alpha = cfg.l1_alphas[l1]
        
        autoencoder = autoencoders[layer][l1]
        model_save_name = cfg.model_name.split("/")[-1]
        save_name = f"base_sae_70m_{layer}_{l1_alpha}"
        torch.save(autoencoder, f"trained_models/base_sae_70m/{save_name}.pt")


# In[ ]:


wandb.finish()


# In[ ]:




