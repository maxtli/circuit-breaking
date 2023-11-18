# %%
from models import import_ablated_model, load_dual_gpt2
import torch
import pickle

# %%
# with open("data/gpt2_means.pkl", "rb") as f:
#     means = pickle.load(f)[0][0]

# model_ablate = import_ablated_model('3', means)

with open('models/preservation_mask_438-7.pkl', 'rb') as f:
    sd = pickle.load(f)

model_ablate = load_dual_gpt2()

model_ablate.load_state_dict(sd)

mask_weights = []
for p in model_ablate.parameters():
    if p.requires_grad:
        mask_weights.append(p)
 
# %%

with open('models/train_record.pkl', 'rb') as f:
    record = pickle.load(f)

# %%
last_rec = record["mask_wts"][-50]
mask_weights = last_rec
# %%
edges = []
for m in mask_weights:
    edges.append(torch.nonzero(m > 0).cpu().numpy())
print(edges)

def convert_to_layered(node):
    if ((node - 1) // 13) + 1 == 0:
        print(node)
    return ((node - 1) // 13) + 1, 0 if node == 0 else (node - 1) % 13

converted_edges = []
output_mask = edges[0]
for edge in output_mask:
    converted_edges.append((convert_to_layered(edge[0]),(13,0)))
i = 1
while i < 13:
    for edge in edges[i*2-1]:
        converted_edges.append((convert_to_layered(edge[0]), (i,edge[1])))
    for edge in edges[i*2]:
        converted_edges.append((convert_to_layered(edge[0]), (i,12)))
    i += 1

# %%
for e in converted_edges:
    (a,b),(c,d) = e
    print(f"({a},{b})->({c},{d})")
# %%
d = model_ablate.state_dict()
for k in d.keys():
    print(d[k])
# %%
ioi_heads = set(['2.2', '4.11', '0.1', '3.0', '0.10', '5.5',\
                 '6.9', '5.8', '5.9', '7.3', '7.9', '8.6',\
                 '8.10', '10.7', '11.0', '9.9', '9.6',\
                 '10.0', '9.0', '9.7', '10.1', '10.2',\
                 '10.6', '10.10', '11.2', '11.9'])

true_positives = 0
false_positives = 0
for e in converted_edges:
    (a,b),(c,d) = e
    a -= 1
    c -= 1
    if f"{a}.{b}" in ioi_heads or f"{a}.{b}" in ioi_heads:
        true_positives += 1
    elif b == 12 or d == 12:
        continue
    false_positives += 1

# %%
print(len(converted_edges))
print(true_positives)
print(false_positives)
# %%
