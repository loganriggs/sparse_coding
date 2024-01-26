#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
# models: "EleutherAI/pythia-6.9b", "usvsnsp/pythia-6.9b-ppo", "lomahony/eleuther-pythia6.9b-hh-sft", "reciprocate/dahoas-gptj-rm-static"
# "EleutherAI/pythia-70m", "lomahony/eleuther-pythia70m-hh-sft", "BlueSunflower/Pythia-70M-chess"
cfg.model_name="EleutherAI/pythia-70m"
cfg.target_name="lomahony/eleuther-pythia70m-hh-sft" #"BlueSunflower/Pythia-70M-chess"
cfg.layers=[0,1,2,3,4,5]
cfg.setting="residual"
# cfg.tensor_name="gpt_neox.layers.{layer}"
cfg.tensor_name="gpt_neox.layers.{layer}" # "gpt_neox.layers.{layer}" (pythia), "transformer.h.{layer}" (rm)
cfg.target_tensor_name="gpt_neox.layers.{layer}"
original_l1_alpha = 2e-3
cfg.l1_alpha=original_l1_alpha
cfg.l1_alphas=[1e-3, 2e-3, 4e-3, 8e-3]
# cfg.l1_alphas=[0, 1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1e-3, 2e-3, 4e-3, 8e-3]
cfg.sparsity=None
cfg.num_epochs=10
cfg.model_batch_size=8 * 8
cfg.lr=1e-2
cfg.kl=False
cfg.reconstruction=False
# cfg.dataset_name="NeelNanda/pile-10k", "BlueSunflower/ChessGames", "Elriggs/openwebtext-100k"
cfg.dataset_name="Elriggs/openwebtext-100k" #"BlueSunflower/ChessGames"
cfg.device="cuda:0"
cfg.ratio = 4
cfg.seed = 0
cfg.max_length = 256
cfg.dataset_lines = None #300_000 # number of lines to download from dataset 300k for 100m tokens
# cfg.device="cpu"


# In[ ]:


tensor_names = [cfg.tensor_name.format(layer=layer) for layer in cfg.layers]
target_tensor_names = [cfg.target_tensor_name.format(layer=layer) for layer in cfg.layers]


# In[ ]:


# Load in the model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
model = model.to(cfg.device)
target_model = AutoModelForCausalLM.from_pretrained(cfg.target_name).to(cfg.device)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)


# In[ ]:


# Download the dataset
# TODO iteratively grab dataset?
token_loader = setup_token_data(cfg, tokenizer, model, seed=cfg.seed, max_lines=cfg.dataset_lines) #cfg.dataset_lines
num_tokens = cfg.max_length*cfg.model_batch_size*len(token_loader)
print(f"Number of tokens: {num_tokens}")


# In[ ]:


# Run 1 datapoint on model to get the activation size
from baukit import Trace, TraceDict

text = "hi buddy"
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


# torch.nn.functional.cosine_similarity(representation, target_representation, dim=-1)
# torch.nn.functional.cosine_similarity(other_representation, target_representation, dim=-1)


# In[ ]:


# Set target sparsity to 10% of activation_size if not set
if cfg.sparsity is None:
    cfg.sparsity = int(activation_size*0.05)
    print(f"Target sparsity: {cfg.sparsity}")

target_lower_sparsity = cfg.sparsity * 0.9
target_upper_sparsity = cfg.sparsity * 1.1
adjustment_factor = 0.1  # You can set this to whatever you like


# In[ ]:


# def autoencoder_name_from_llm_name(llm_name):
#     if llm_name == "EleutherAI/pythia-6.9b":
#         return "base_sae_6b"
#     if llm_name == "EleutherAI/pythia-70m":
#         return "base_sae_70m"
#     if llm_name == "lomahony/eleuther-pythia6.9b-hh-sft":
#         return "sft_sae_6b"
#     if llm_name == "lomahony/eleuther-pythia70m-hh-sft":
#         return "sft_sae_70m"
#     if llm_name == "usvsnsp/pythia-6.9b-ppo":
#         return "ppo_sae_6b"
#     if llm_name == "reciprocate/dahoas-gptj-rm-static":
#         return "rm_sae_gptj"
#     return "Error"

def autoencoder_name_from_llm_name(llm_name, layer, l1_alpha):
    # model_save_name = llm_name.split("/")[-1]
    return f"base_sae_70m_{layer}_{l1_alpha}"
    # return f"{model_save_name}_{layer}_{l1_alpha}"
    
def autoTED_name_from_llm_name(llm_name, mode, layer, l1_alpha):
    # if llm_name == "EleutherAI/pythia-6.9b":
    #     return "base_autoTED_6b"
    if llm_name == "EleutherAI/pythia-70m":
        return f"base_autoTED_70m/base_autoTED_70m_{mode}_{layer}_{l1_alpha}"
    # if llm_name == "lomahony/eleuther-pythia6.9b-hh-sft":
    #     return "sft_autoTED_6b"
    # if llm_name == "lomahony/eleuther-pythia70m-hh-sft":
    #     return "sft_autoTED_70m"
    # if llm_name == "usvsnsp/pythia-6.9b-ppo":
    #     return "ppo_autoTED_6b"
    # if llm_name == "reciprocate/dahoas-gptj-rm-static":
    #     return "rm_autoTED_gptj"
    return "Error"


# In[ ]:


# load autoencoder
from autoencoders.learned_dict import TiedSAE, UntiedSAE, AnthropicSAE, TransferSAE
from torch import nn

model_save_name = cfg.model_name.split("/")[-1]
print(f"Loading autoTEDs from model {model_save_name}")


modes = ["scale", "rotation", "bias", "free"]
autoTEDs = []
for layer in cfg.layers:
    l1_variants = []
    
    for l1 in cfg.l1_alphas:
        mode_autoTEDS = []
        for mode in modes:
            save_name = autoTED_name_from_llm_name(cfg.model_name, mode, layer, l1)
            autoencoder = torch.load(f"trained_models/{save_name}.pt")
            autoencoder.to_device(cfg.device)
            mode_autoTEDS.append(autoencoder)
            
        l1_variants.append(mode_autoTEDS)
        
    autoTEDs.append(l1_variants)


# In[ ]:


# Initialize New transfer autoencoder
from autoencoders.learned_dict import TiedSAE, UntiedSAE, AnthropicSAE, TransferSAE
from torch import nn

# modes = ["scale", "rotation", "bias", "free"]
transfer_autoencoders = []
optimizers = []
for layer in cfg.layers:
    l1_variants = []
    l1_optimizers = []
    
    for l1 in range(len(cfg.l1_alphas)):
        
        mode_tsaes = []
        mode_opts = []
        for mode_id, mode in enumerate(modes):
            autoencoder = autoTEDs[layer][l1][mode_id]
            mode_tsae = TransferSAE(
                # n_feats = n_dict_components, 
                # activation_size=activation_size,
                autoencoder,
                decoder=autoencoder.get_learned_dict().detach().clone(),
                decoder_bias=autoencoder.shift_bias.detach().clone(),
                scale=autoencoder.scale.detach().clone(),
                mode=mode,
            )
            mode_tsae.set_grad()
            mode_tsaes.append(mode_tsae)
            mode_opts.append(
                torch.optim.Adam(mode_tsae.parameters(), lr=cfg.lr)
            )
        
        l1_variants.append(mode_tsaes)
        l1_optimizers.append(mode_opts)
    transfer_autoencoders.append(l1_variants)
    optimizers.append(l1_optimizers)

# reset autoencoders list to just include free autoTEDs
autoencoders = [[autoTEDs[layer][l1][-1] for l1 in range(len(cfg.l1_alphas))] for layer in cfg.layers ]


# In[ ]:


# Wandb setup
secrets = json.load(open("secrets.json"))
wandb.login(key=secrets["wandb_key"])
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb_run_name = f"{cfg.model_name}_transfer_{start_time[4:]}_{cfg.sparsity}"  # trim year
print(f"wandb_run_name: {wandb_run_name}")
wandb.init(project="sparse coding", config=dict(cfg), name=wandb_run_name)


# In[ ]:


def training_step(autoencoder, base_activation, target_activation, transfer_autoencoders, optimizers, step_number, dead_features, log_every=100, log_prefix="", indiv_prefixes = None):
    i = step_number
    c = autoencoder.encode(base_activation.to(cfg.device))
    x_hat = autoencoder.decode(c)
    
    autoencoder_loss = (x_hat - target_activation.to(cfg.device)).pow(2).mean()
    dead_features += c.sum(dim=0).cpu()
    
    wandb_log = {}
    
    if not isinstance(transfer_autoencoders, list):
        transfer_autoencoders = [transfer_autoencoders]
        optimizers = [optimizers]
        
    if not isinstance(indiv_prefixes, list):
        indiv_prefixes = [indiv_prefixes]
    
    for tsae, optimizer, indiv_prefix in zip(transfer_autoencoders, optimizers, indiv_prefixes):
        x_hat = tsae.decode(c)
        
        reconstruction_loss = (x_hat - target_activation.to(cfg.device)).pow(2).mean()
        total_loss = reconstruction_loss # NO L1 LOSS

        if (i % log_every == 0): # Check here so first check is model w/o change
            # self_similarity = torch.cosine_similarity(tsae.decoder, last_decoders[mode], dim=-1).mean().cpu().item()
            # last_decoders[mode] = tsae.decoder.clone().detach()
            num_tokens_so_far = i*cfg.max_length*cfg.model_batch_size
            with torch.no_grad():
                sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
            # print(f"{log_prefix}Reconstruction Loss: {reconstruction_loss:.2f} | Tokens: {num_tokens_so_far}") # | Self Similarity: {self_similarity:.2f}")
            wandb_log.update({
                f'{log_prefix}{indiv_prefix}Reconstruction Loss': reconstruction_loss.item(),
                # f'{log_prefix}{mode} Self Similarity': self_similarity
            })

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    if (i % log_every == 0):
        with torch.no_grad():
            sparsity = (c != 0).float().mean(dim=0).sum().cpu().item()
            # Count number of dead_features are zero
            num_dead_features = (dead_features == 0).sum().item()
        # print(f"{log_prefix}Sparsity: {sparsity:.1f} | Dead Features: {num_dead_features} | Reconstruction Loss: {autoencoder_loss:.2f} | Tokens: {num_tokens_so_far}")
        # dead_features = torch.zeros(autoencoder.encoder.shape[0])
        wandb_log.update({
                f'{log_prefix}SAE Sparsity': sparsity,
                f'{log_prefix}Dead Features': num_dead_features,
                f'{log_prefix}SAE Reconstruction Loss': autoencoder_loss.item(),
                f'{log_prefix}Tokens': num_tokens_so_far,
            })
        
        # wandb.log(wandb_log)
    
    binary_activations = (c != 0).sum(dim=0).cpu().detach()
    return wandb_log, dead_features, binary_activations


# In[ ]:


# Training transfer autoencoder
# token_loader = setup_token_data(cfg, tokenizer, model, seed=cfg.seed, max_lines=cfg.dataset_lines)
dead_features = [[torch.zeros(autoencoder.encoder.shape[0])
                         for l1 in cfg.l1_alphas]
                         for layer in range(len(tensor_names))]
frequency = [[torch.zeros(autoencoder.encoder.shape[0])
                         for l1 in cfg.l1_alphas]
                         for layer in range(len(tensor_names))]
# auto_dead_features = torch.zeros(autoencoder.encoder.shape[0])

max_num_tokens = 100_000_000
log_every=100
# Freeze model parameters 
model = model.to(cfg.device)
model.eval()
target_model = target_model.to(cfg.device)
target_model.eval()


# last_decoders = dict([(modes[i],transfer_autoencoders[i].decoder.clone().detach()) for i in range(len(transfer_autoencoders))])
model_on_gpu = True

saved_inputs = []
i = 0 # counts all optimization steps
num_saved_so_far = 0
print("starting loop")
# for (base_activation, target_activation) in tqdm(generate_activations(model, token_loader, cfg, model_on_gpu=model_on_gpu, num_batches=500), 
#                                                  total=int(max_num_tokens/(cfg.max_length*cfg.model_batch_size))):
tqdm_length = min(int(max_num_tokens/(cfg.max_length*cfg.model_batch_size)), len(token_loader))
for i, batch in enumerate(tqdm(token_loader,total=tqdm_length)):
    tokens = batch["input_ids"].to(cfg.device)
    with torch.no_grad(): # As long as not doing KL divergence, don't need gradients for model       
        base_activations = []
        with TraceDict(model, tensor_names) as ret:
            _ = model(tokens)
            s = ret[tensor_names[0]]
            for layer_name in tensor_names:
                representation = ret[layer_name].output[0]
                flattened_rep = rearrange(representation, "b seq d_model -> (b seq) d_model")
                base_activations.append(flattened_rep)
                
        target_activations = []
        with TraceDict(target_model, tensor_names) as ret:
            _ = target_model(tokens)
            for layer_name in tensor_names:
                representation = ret[layer_name].output[0]
                flattened_rep = rearrange(representation, "b seq d_model -> (b seq) d_model")
                target_activations.append(flattened_rep)
    
    wandb_log = {}
    for layer in range(len(cfg.layers)):
        for l1_id in range(len(cfg.l1_alphas)):
            indiv_prefixes = [f"{layer} {cfg.l1_alphas[l1_id]} {mode} " for mode in modes]
            wandb_logging, dead_features[layer][l1_id], binary_acts = training_step(autoencoders[layer][l1_id], base_activations[layer], target_activations[layer], 
                                                                        transfer_autoencoders[layer][l1_id], optimizers[layer][l1_id], i, 
                                                                        dead_features[layer][l1_id], log_every, log_prefix=f"{layer} {cfg.l1_alphas[l1_id]} ",
                                                                        indiv_prefixes=indiv_prefixes)
            wandb_log.update(wandb_logging)
            frequency[layer][l1_id] += binary_acts
    
    if len(wandb_log) > 0:
        wandb.log(wandb_log)
        pass
        
    i+=1
    
    # if ((i+2) % 6000==0): # save periodically but before big changes
    #     for layer in range(len(cfg.layers)):
    #         for l1_id in range(len(cfg.l1_alphas)):
    #             for mode_id in range(len(modes)):
    #                 mode = modes[mode_id]
    #                 l1_alpha = cfg.l1_alphas[l1_id]
    #                 model_save_name = cfg.model_name.split("/")[-1]
    #                 save_name = f"transfer_base_sft_70m_{mode}_{layer}_{l1_alpha}_ckpt{num_saved_so_far}" 

    #                 # Make directory trained_models if it doesn't exist
    #                 import os
    #                 if not os.path.exists("trained_models"):
    #                     os.makedirs("trained_models")
    #                 # Save model
    #                 torch.save(transfer_autoencoders[layer][l1_id][mode_id], f"trained_models/new_transfer/{save_name}.pt")
    #                 # torch.save(dead_features[layer][l1_id], f"trained_models/dead_features_{model_save_name}_{layer}_{l1_alpha}.pt")
        
    #     num_saved_so_far += 1
                
    
    num_tokens_so_far = i*cfg.max_length*cfg.model_batch_size
    if(num_tokens_so_far > max_num_tokens):
        print(f"Reached max number of tokens: {max_num_tokens}")
        break
    
for layer in range(len(cfg.layers)):
    for l1_id in range(len(cfg.l1_alphas)):
        frequency[layer][l1_id] = frequency[layer][l1_id]/num_tokens_so_far


# In[ ]:


# Save model at end
    
for layer in range(len(cfg.layers)):
    for l1_id in range(len(cfg.l1_alphas)):
        for mode_id in range(len(modes)):
            mode = modes[mode_id]
            l1_alpha = cfg.l1_alphas[l1_id]
            model_save_name = cfg.model_name.split("/")[-1]
            save_name = f"transfer_base_sft_70m_{mode}_{layer}_{l1_alpha}" 

            # Make directory trained_models if it doesn't exist
            import os
            if not os.path.exists("trained_models"):
                os.makedirs("trained_models")
            # Save model
            torch.save(transfer_autoencoders[layer][l1_id][mode_id], f"trained_models/new_transfer/{save_name}.pt")
            # torch.save(dead_features[layer][l1_id], f"trained_models/dead_features_{model_save_name}_{layer}_{l1_alpha}.pt")


# In[ ]:


wandb.finish()


# In[ ]:




