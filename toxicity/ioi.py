# %%
import datasets
import re
import torch
import math
from itertools import cycle
from data import retrieve_owt_data
from models import tokenizer, load_demo_gpt2, load_dual_gpt2
from torch.utils.data import DataLoader
import pickle
import numpy as np
from inference import infer_batch_with_owt, infer_batch, prepare_fixed_demo
from torch.optim import AdamW
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
# %%

device = "cuda:0" if torch.cuda.is_available else "cpu"

# ioi_ds = datasets.load_from_disk("../../plausibleablation/data/ioi/ioi")

from ioi_dataset import IOIDataset, format_prompt, make_table

# %%
N = 10000
clean_dataset = IOIDataset(
    prompt_type='mixed',
    N=N,
    tokenizer=tokenizer,
    prepend_bos=False,
    seed=1,
    device=device
)
corr_dataset = clean_dataset.gen_flipped_prompts('ABC->XYZ, BAB->XYZ')

# %%

# %%
# context_length = CONTEXT_LENGTH

# toxic_data_loader = retrieve_toxic_data(toxic_batch_size, context_length, tokenizer)
# # toxic_data_loader = retrieve_toxic_filtered_data(toxic_batch_size)
# owt_data_loader = retrieve_owt_data(owt_batch_size)

# with open("data/gpt2_means.pkl", "rb") as f:
#     means = pickle.load(f)[0][0]

# model = load_demo_gpt2(means=means)

model = load_dual_gpt2()
model.train()

make_table(
  colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
  cols = [
    map(format_prompt, clean_dataset.sentences),
    tokenizer.decode(clean_dataset.s_tokenIDs).split(),
    tokenizer.decode(clean_dataset.io_tokenIDs).split(),
    map(format_prompt, corr_dataset.sentences),
  ],
  title = "Sentences from IOI vs ABC distribution",
)

# %%
epochs_left = 1
lr = .01 # free
weight_decay = 0
threshold = 0.5
epochs_trained = 0
batches_trained = 0

batch_size = 8
max_context_length = 35

mask_params = []
param_names = []
for name, p in model.named_parameters():
    if p.requires_grad:
        param_names.append(name)
        mask_params.append(p)
optimizer = AdamW(mask_params, lr=lr, weight_decay=weight_decay)

demos = prepare_fixed_demo(tokenizer, batch_size, demo="")
edge_threshold = 100
criterion = torch.nn.CrossEntropyLoss()
kl_loss = torch.nn.KLDivLoss()

# %%
epochs_left = 20
clamp_every = 200 # how often to randomly sample edges
log_every = 200 # how often to log
option_every = 100000 # how often to ask whether to stop
starting_batch = 0 # when pausing and resuming training so certain batches in the cycle do not get over-represented


# %%
noise_every = 20
sigma = .01
# %%
record_batches = 50

records = {"ioi_loss": [], "kept_edges": [], "reg_penalty": [], "mask_wts": []}
tmp_records = {"ioi_loss": [], "kept_edges": [], "reg_penalty": []}
# %%
# epochs_left = 1
while epochs_left > 0:
    for e in range(epochs_left):
        batch_count = math.ceil(clean_dataset.toks.shape[0] / batch_size)

        for c in tqdm(range(batch_count)):

            if c < starting_batch:
                continue
            if c == starting_batch:
                starting_batch = 0

            start_batch = c * batch_size
            end_batch = (c+1) * batch_size

            clean_toks = clean_dataset.toks[start_batch:end_batch]
            corr_toks = corr_dataset.toks[start_batch:end_batch]

            last_token_pos = ((clean_toks != tokenizer.pad_token_id) * torch.arange(clean_toks.shape[1]).to(device)).argmax(dim=-1).unsqueeze(1)
            last_token_labels = clean_toks.gather(1,last_token_pos).squeeze(1)
            last_token_corr_labels = corr_toks.gather(1,last_token_pos).squeeze(1)

            max_seq_len = last_token_pos.max().item()
            clean_toks = clean_toks[:,:max_seq_len]
            corr_toks = clean_toks[:,:max_seq_len]

            total_preserving = 0
            kept_edges = 0
            penalty = 0
            for p in mask_params:
                total_preserving += p.sum()
                kept_edges += p[p.data > 0.5].shape[0]
                penalty += p.sum() * math.sqrt(batches_trained) / 2000 # why 2000? free
            
            logits, corr_logits = model(clean_toks, corr_toks) # 0 is the logits

            infer_token_pos = last_token_pos.unsqueeze(2).repeat(1,1,logits.shape[2]) - 1
            last_token_logits = logits.gather(1,infer_token_pos).squeeze(1)
            corr_last_token_logits = corr_logits.gather(1,infer_token_pos).squeeze(1)

            ioi_loss = criterion(last_token_logits, last_token_labels)

            loss = penalty + ioi_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tmp_records["ioi_loss"].append(ioi_loss.item())
            tmp_records["kept_edges"].append(kept_edges)
            tmp_records["reg_penalty"].append(penalty.item())
            batches_trained += 1

            if noise_every > 0 and batches_trained % noise_every == 0:
                for p in mask_params:
                    p.data += torch.normal(0,sigma,p.data.shape).to(device)
            for p in mask_params:
                p.data.clamp_(0,1)

            if batches_trained % record_batches == 0:
                for k in tmp_records:
                    records[k].append(np.mean(tmp_records[k]).item())
                    tmp_records[k] = []
                records["mask_wts"].append(([p.data.detach() for p in mask_params]))
                        
            if batches_trained % clamp_every == 0:
                # ablated_edges = 0
                all_weights = []
                for p in mask_params:
                    all_weights.append(p.data.flatten().cpu())

                    # set the edge to 0 or 1 with probability being its current wt
                    rand_sample = torch.rand(p.data.shape).to(device)
                    p.data[p.data < rand_sample] = 0
                    p.data[p.data > rand_sample] = 1
                    
                    # p.data[p.data < threshold] = 0
                    # p.data[p.data >= threshold] = 1
                    # ablated_edges += p[p.data < 0.5].shape[0]
                all_weights = torch.cat(all_weights).numpy()
                ax = sns.histplot(all_weights[all_weights > .05])
                plt.show()
            if batches_trained % log_every == 0:
                # corr_loss = kl_loss(last_token_logits, corr_last_token_logits)
                print("Epochs trained: ", epochs_trained)
                print(f"Loss: {loss.item():.4f}")
                print(f"Total preserved: {total_preserving:.4f}")
                print("Edges kept: ", kept_edges)
                print("Clean loss: ", ioi_loss.item())
                # print("Corr loss: ", corr_loss.item())
                print("Penalty: ", penalty)
                print("\n")
            
            if batches_trained % option_every == 0:
                if input(f"stop? edges: {kept_edges}, clean_loss: {ioi_loss.item()}") == 'y':
                    starting_batch = c+1
                    break
        epochs_trained += 1                
        if kept_edges < edge_threshold:
            break
        prev_params = mask_params
    try:
        epochs_left = int(input('continue training for this number of epochs: '))
    except:
        epochs_left = 0
    if epochs_left > 0:
        try:
            clamp_every = int(input(f"set CLAMP (sampled) frequency, currently {clamp_every}"))
            clamp_every = int(input(f"set NOISE frequency, currently {noise_every}"))
            clamp_every = int(input(f"set NOISE amount (std), currently {sigma}"))
            log_every = int(input(f"set LOG frequency, currently {log_every}"))
            option_every = int(input(f"set OPTION frequency, currently {option_every}"))
            edge_threshold = int(input(f"set edge threshold, currently {edge_threshold}"))
        except:
            pass



# %%
with open("models/train_record_2.pkl", "wb") as f:
    pickle.dump(records, f)

# %%
with open("models/preservation_mask_438-7.pkl", "wb") as f:
    pickle.dump(model.state_dict(), f)

# %%
from transformer import Config, LayerNorm, Attention, MLP, Embed, PosEmbed, Unembed
import einops
import torch.nn as nn
from fancy_einsum import einsum


# %%

"""## Transformer Block"""
class TransformerBlock(nn.Module):
    def __init__(self, cfg, prev_layers: int):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

        for p in self.parameters():
            p.requires_grad = False

        prev_nodes = (cfg.n_heads + 1) * prev_layers + 1
        edge_mask_attentions_init = torch.ones((prev_nodes, cfg.n_heads))
        self.edge_mask_attentions = torch.nn.Parameter(edge_mask_attentions_init, requires_grad=True)

        edge_mask_mlp_init = torch.ones((prev_nodes + cfg.n_heads, ))
        self.edge_mask_mlp = torch.nn.Parameter(edge_mask_mlp_init, requires_grad=True)

    def forward(self, resid_pre, means=False):

        # resid_pre [batch, position, d_model, prev_head_idx]
        masked_residuals = einsum("batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model", resid_pre, self.edge_mask_attentions)
        if isinstance(means, torch.Tensor):
            if means.dim() == 2:
                means = means.unsqueeze(0,1)
            
            assert means.dim() == 4

            masked_means = einsum("b p prev_head_idx d_model, prev_head_idx n_heads -> b p n_heads d_model", means, 1 - self.edge_mask_attentions)
            masked_residuals = masked_residuals + masked_means

        # print(self.edge_mask_attentions)
        # torch.sum(masked_residuals, dim=2, keepdim=True)

        normalized_resid_pre = self.ln1(masked_residuals, parallel=True)
        # print(normalized_resid_pre[:,:,0])
        # print(torch.allclose(normalized_resid_pre[:,:,torch.randperm(normalized_resid_pre.shape[2])],normalized_resid_pre))

        attn_out = self.attn(normalized_resid_pre)

        # self.saved_output = attn_out

        residual = torch.cat((resid_pre, attn_out), dim=2)

        masked_mlp_residual = einsum("batch position prev_head_idx d_model, prev_head_idx -> batch position d_model", residual, self.edge_mask_mlp)
        
        normalized_resid_mid = self.ln2(masked_mlp_residual)
        mlp_out = self.mlp(normalized_resid_mid)
        mlp_out = einops.rearrange(mlp_out, "batch position d_model -> batch position 1 d_model")

        residual = torch.cat((residual, mlp_out), dim=2)

        return residual

class DualTransformer(nn.Module):
    def __init__(self, cfg, init_mask=False):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
        for p in self.parameters():
            p.requires_grad = False

        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        total_nodes = (cfg.n_heads + 1) * cfg.n_layers + 1

        if init_mask is False:
            init_mask = torch.ones((total_nodes,))
        self.output_mask = torch.nn.Parameter(init_mask, requires_grad=True)
    
    def forward(self, tokens, corr_tokens, return_states=False):
        # tokens [batch, position]
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)

        residual = embed + pos_embed
        residual = einops.rearrange(residual, "batch position d_model -> batch position 1 d_model")
        
        corr_embed = self.embed(corr_tokens)
        corr_pos_embed = self.pos_embed(corr_tokens)
        
        corr_residual = corr_embed + corr_pos_embed
        corr_residual = einops.rearrange(corr_residual, "batch position d_model -> batch position 1 d_model")

        for i, block in enumerate(self.blocks):
            # print(i)
            residual = block(residual, corr_residual)
            corr_residual = block(corr_residual, False)
            # if hasattr(self,"saved_states"):
            #     self.saved_states = torch.cat((self.saved_states, block.saved_output.unsqueeze(0)), dim=0)
            # else:
            #     self.saved_states = block.saved_output.unsqueeze(0)
        
        if return_states:
            return residual
        
        residual = einsum("batch position prev_head_idx d_model, prev_head_idx -> batch position d_model", residual, self.output_mask)
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        # logits have shape [batch, position, logits]
        # with open("saved_states_new.pkl", "wb") as f:
        #     pickle.dump(self.saved_states, f)
        return [logits]


def load_dual_gpt2():
    with open("models/gpt2_weights.pkl", "rb") as f:
        gpt2_weights = pickle.load(f)
    demo_gpt2 = DualTransformer(Config(debug=False))
    demo_gpt2.load_state_dict(gpt2_weights, strict=False)
    demo_gpt2.cuda()
    return demo_gpt2

# %%

model = load_dual_gpt2()



# %%

data_loader = DataLoader(ioi_ds['train'], batch_size=batch_size, shuffle=True, pin_memory=True)

owt_loader = retrieve_owt_data(batch_size, max_context_length, tokenizer)
kl_loss = torch.nn.KLDivLoss()
owt_iter = cycle(owt_loader)

# %%
# %%
batch = tokenizer(next(iter(data_loader))['ioi_sentences'], padding=True, return_tensors='pt')['input_ids'].to(device)
last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(device)).argmax(dim=-1) 


# %%
    if find_last_token:
        # full sequence includes the IO
        last_token_pos = ((batch != tokenizer.pad_token_id) * torch.arange(batch.shape[1]).to(device)).argmax(dim=-1) - 1
    else:
        last_token_pos = -1 * torch.ones(batch.shape[0]).to(device)

# %%

prev_params = None

while epochs_left > 0:
    for e in tqdm(range(epochs_left)):
        for c, batch in enumerate(toxic_data_loader):
            total_preserving = 0
            ablated_edges = 0
            penalty = 0
            for p in mask_params:
                total_preserving += p.sum()
                ablated_edges += p[p.data < 0.5].shape[0]
                penalty += max(0, p.sum() * (epochs_trained-20) / 10000) # why 2000? free

            # demos = batch[:, :FILTER_DEMO_LEN]
            # completions = batch[:, FILTER_DEMO_LEN:]

            ioi_loss = infer_batch(model, criterion, batch_size, demos)
            loss = penalty + alpha * ioi_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            for p in mask_params:
                p.data.clamp_(0,1)
        epochs_trained += 1
        if epochs_trained % clamp_every == 0:
            ablated_edges = 0
            for p in mask_params:
                p.data[p.data < threshold] = 0
                p.data[p.data >= threshold] = 1
                ablated_edges += p[p.data < 0.5].shape[0]
        if epochs_trained % log_every == 0:
            print("Epochs trained: ", epochs_trained)
            print(f"Loss: {loss.item():.4f}")
            print(f"Total preserved: {total_preserving:.4f}")
            print("Edges ablated: ", ablated_edges)
            print("Toxic loss: ", tox_loss.item())
            print("OWT loss: ", owt_loss.item())
            print("Penalty: ", penalty)
            if input('evaluate? (y)') == 'y':
                evaluate_model(model, toxic_batches=1, owt_batches=1)
            print("\n")
                
        if epochs_trained > 50 and ablated_edges < edge_threshold:
            break
        prev_params = mask_params
    epochs_left = int(input('continue training for this number of epochs: '))
    log_every = int(input('set log frequency'))
    edge_threshold = int(input('set edge threshold'))

# %%

total_preserving = 0
for p in mask_params:
    p.data[p.data < threshold] = 0
    p.data[p.data >= threshold] = 1
    total_preserving += p.data.sum()
print(total_preserving)

# %%
state_dict = model.state_dict()
for name, param in zip(param_names, prev_params):
    state_dict[name] = param

# %%
with open("models/masked_gpt2_mean_ablation_v6.pkl", "wb") as f:
    pickle.dump(model.state_dict, f)

# %%
"""We can now plot a loss curve!"""

# px.line(y=losses, x=np.arange(len(losses))*(model.cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")