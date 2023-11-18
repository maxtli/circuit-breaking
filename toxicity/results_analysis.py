# %%

import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pandas as pd
# %%

# with open("models/train_record.pkl", "rb") as f:
#     train_record = pickle.load(f)

with open("models/train_record_2.pkl", "rb") as f:
    train_record_2 = pickle.load(f)

# %%
idxes = [i for i in range(len(train_record_2["ioi_loss"]))]

# %%
ioi_loss = pd.Series(train_record_2["ioi_loss"]).rolling(50).mean()

# %%
ax1 = sns.lineplot(x=idxes, y=ioi_loss.fillna(0),ci=None, label="ioi_loss")
ax2 = sns.lineplot(x=idxes, y=train_record_2["kept_edges"], ax=ax1.twinx(), ci=None, color="red", label="kept_edges")
ax3 = sns.lineplot(x=idxes, y=train_record_2["reg_penalty"], ax=ax1.twinx(), ci=None, color="orange", label="reg_penalty")

ax1.legend()
ax2.legend()
ax3.legend()
# %%
