
"""
Computes the mean of the GPT2 embeddings for the OWT dataset.
"""

# %%

from data import retrieve_owt_data
from models import load_demo_gpt2, tokenizer, DEVICE
from tqdm import tqdm
import torch
import ioi_dataset

# %%
batch_size = 50
ctx_length = 50
model = load_demo_gpt2(means=False)

N = 10000
clean_dataset = ioi_dataset.IOIDataset(
    prompt_type='mixed',
    N=N,
    tokenizer=tokenizer,
    prepend_bos=False,
    seed=1,
    device=DEVICE
)
corr_dataset = clean_dataset.gen_flipped_prompts('ABC->XYZ, BAB->XYZ')


# data_loader = retrieve_owt_data(batch_size, ctx_length, tokenizer)

# %%

def compute_means(data_loader):
    means = []
    meta_means = []
    for c, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            means.append(model(batch['tokens'].long(), return_states=True).mean(dim=[0,1],keepdim=True))
        if c % 50 == 0:
            meta_means.append(torch.stack(means, dim=0).mean(dim=0))
            means = []
    all_means = torch.stack(meta_means, dim=0).mean(dim=0)
    return all_means

means = compute_means(data_loader)

# %%

import pickle 
with open('data/gpt2_means.pkl', 'wb') as f:
    pickle.dump(means, f)

# %%
