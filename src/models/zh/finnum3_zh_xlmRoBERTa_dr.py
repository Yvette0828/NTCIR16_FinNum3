from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

import torch
from transformers import *
from fastai.text.all import *

from blurr.data.all import *
from blurr.modeling.all import *

trn_json_f = open('/root/NTCIR_FinNum3/ZH/Dataset/FinNum-3_train.json')
trn_json = json.load(trn_json_f)
trn_json_f.close()

"""# EDA for FinNum-2 Training Set

## ETL
"""

text_catgy = defaultdict(lambda: defaultdict(list))
category_ = Counter()
claim_num = Counter()
for d in trn_json:
    text = d['text']
    category = d['category']
    del d['text']
    del d['category']
    text_catgy[text][category].append(d)
    category_[category] += 1
    claim_num[d['claim']] += 1

"""# Preprocess"""

SEED = 42

def reset_randomness(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # This implies torch.cuda.manual_seed_all(SEED) now
    # if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True  # About 15% slower but...
    torch.backends.cudnn.benchmark = False

def insert_tags(df):
    df = df.assign(
        tagged_text=df[['text', 'offset']].apply(
            lambda x: x[0][:x[1]] + ' xxnum ' + x[0][x[1]:], axis=1
        )
    )
    # df.tagged_text = df[['tagged_text', 'category']].apply(
    #     lambda x: re.sub(f'(\$({x[1]}))', r'$ xxtag \2', x[0]), axis=1
    # )
    df.tagged_text = '<s> ' + df.tagged_text + ' </s>'
    return df

def yield_double_redaction(
    txt,
    untouchable_token_set,
    redact_change=0.2,
    unk_chance=0.2,
    mask='<mask>',
    unk='<unk>',
    token_delimiter=' ',
    seed=SEED
):
    random.seed(seed)
    chosen_idx_set_list = []
    tkns = txt.split(token_delimiter)
    tkn_idx_set = {
        tkn_idx for tkn_idx, tkn in enumerate(tkns)
        if tkn not in untouchable_token_set
    }
    n_tkn = len(tkn_idx_set)
    n_sample = math.ceil(n_tkn * redact_change)
    while n_sample <= n_tkn:
        chosen_idx_set = set(random.sample(sorted(tkn_idx_set), k=n_sample))
        chosen_idx_set_list += [chosen_idx_set]
        tkn_idx_set -= chosen_idx_set
        n_tkn = len(tkn_idx_set)
    for idx_set in chosen_idx_set_list:
        dr_tkns = tkns.copy()
        for chosen_idx in idx_set:
            noise = unk if random.random() < unk_chance else mask
            dr_tkns[chosen_idx] = noise
        yield token_delimiter.join(dr_tkns)


def double_redact_tagged_finnum_df(df, special_token_set):
    duplicated_rows = []
    for n, row in df.iterrows():
        for redacted_tagged_text in yield_double_redaction(
            row['tagged_text'],
            {row['target_numeral'], row['category']} | special_token_set, 
        ):
            row['tagged_text'] = redacted_tagged_text
            duplicated_rows += [row.copy()]
    return pd.DataFrame(duplicated_rows)

trn_df = pd.read_json('/root/NTCIR_FinNum3/ZH/Dataset/FinNum-3_train.json')
trn_df = insert_tags(trn_df)
trn_df = trn_df.assign(is_vld=False)
dr_trn_df = double_redact_tagged_finnum_df(
    trn_df, {'xxtag', 'xxnum', '<s>', '</s>'})

vld_df = pd.read_json('/root/NTCIR_FinNum3/ZH/Dataset/FinNum-3_dev.json')
vld_df = insert_tags(vld_df)
vld_df = vld_df.assign(is_vld=True)

#test set no category  
tst_df = pd.read_json('/root/NTCIR_FinNum3/ZH/Dataset/FinNum-3_test.json')
tst_df = insert_tags(tst_df)

"""# Train"""

model_cls = AutoModelForSequenceClassification
pretrained_model_name = 'xlm-roberta-base'  #'hfl/chinese-macbert-large'

(
    hf_arch,
    hf_config,
    hf_tokenizer,
    hf_model
) = BLURR.get_hf_objects(
    pretrained_model_name,
    model_cls=model_cls,
    config_kwargs={'num_labels': 2}
)

hf_tokenizer.add_special_tokens({
    'additional_special_tokens': ['xxtag', 'xxnum']
})
hf_tokenizer.sanitize_special_tokens()
hf_model.resize_token_embeddings(len(hf_tokenizer))

trn_bs = 8

# %%timeit -n 1 -r 1 global trn_bs, dls
reset_randomness()

dblock = DataBlock(
    blocks=(
        HF_TextBlock(hf_arch=hf_arch, hf_tokenizer=hf_tokenizer, hf_config =hf_config,hf_model =hf_model, max_length = 512, trunction = True, padding = True),
        CategoryBlock),
    get_x=ColReader('tagged_text'),
    get_y=ColReader('claim'),
    splitter=ColSplitter(col='is_vld')
)

dls = dblock.dataloaders(
    trn_df.append(vld_df), bs=trn_bs, val_bs=256, num_workers=0

)

dls_dr = dblock.dataloaders(
    dr_trn_df.append(vld_df), bs=trn_bs, val_bs=256, num_workers=0

)

torch.save(dls, '/root/NTCIR_FinNum3/ZH/Model/FinNum3_train_ZH_xlmRoBERTa.pth')
torch.save(dls, '/root/NTCIR_FinNum3/ZH/Model/FinNum3_dr_train_ZH_xlmRoBERTa.pth')


torch.save(
    dls.test_dl(tst_df.tagged_text.tolist(), bs=256),
    '/root/NTCIR_FinNum3/ZH/Model/FinNum3_test.pth'
)

# torch.save(
#     dblock.dataloaders(
#         dr_trn_df.append(vld_df), bs=trn_bs, val_bs=256, num_workers=0
#     ),
#     '/content/drive/MyDrive/NTPU/NTCIR/FinNum3/FinNum2_train_ZH_xlmRoBERTa.pth'
# )

class_weights = torch.FloatTensor([0.31, 1]).cuda() # 999/3220 (out/in)
lrnr = Learner(
    torch.load('/root/NTCIR_FinNum3/ZH/Model/FinNum3_train_ZH_xlmRoBERTa.pth'),
    HF_BaseModelWrapper(hf_model),
    opt_func=partial(Lamb, decouple_wd=True),
    loss_func=CrossEntropyLossFlat(weight=class_weights), #BCEWithLogitsLossFlat
    metrics=[
        accuracy,
        F1Score(average='micro'),
        F1Score(average='macro'),
        MatthewsCorrCoef(),
        CohenKappa(weights='linear'),
        Jaccard(average='weighted'),
    ],
    cbs=[HF_BaseModelCallback],
    splitter=hf_splitter,
    path='/root/NTCIR_FinNum3/ZH/Model',
    moms = (0.8, 0.7, 0.8)
)
lrnr = lrnr.to_fp16()
lrnr.create_opt()

# lrnr.freeze()

# reset_randomness()
# print(lrnr.lr_find(suggest_funcs=(minimum, steep, valley, slide)))
# lrnr.lr_find(suggestions=True)
# lrnr.lr_find(start_lr=2e-8, end_lr=2e-2, suggestions=True)

#####

reset_randomness()
lr = 2.8e-4
lrnr.fit_one_cycle(2, lr_max = slice(4e-06, lr))

lr2 = 1.4e-04  # lr/2
lrnr.dls = dls_dr
lrnr.fit_one_cycle(3, lr_max = slice(2e-06, lr2))

lr3 = 7e-05  # lr2/2
lrnr.dls = dls
lrnr.fit_one_cycle(1, lr_max = slice(1e-06, lr3))

lr4 = 1.4e-05  # lr3/5
lrnr.fit_one_cycle(1, lr_max = slice(2e-07, lr4))

# lrnr.unfreeze()

# lrnr.save(f'fn_cf-b{trn_bs}-frozen-cl1_lr5En3-cl1_lr5En3')

# reset_randomness()

_, _, preds = lrnr.get_preds(with_decoded=True)
vld_df.assign(prediction=preds.tolist())[[
    'text', 'target_numeral', 'offset', 'category', 'claim', 'prediction'
]].to_json(
    '/root/NTCIR_FinNum3/ZH/Result/FinNum3_zh_dev_dr.json',
    orient='records')

tst_dl = torch.load('/root/NTCIR_FinNum3/ZH/Model/FinNum3_test.pth')
# reset_randomness()
_, _, preds = lrnr.get_preds(dl=tst_dl, with_decoded=True)
tst_df.assign(prediction=preds.tolist())[[
    'text', 'target_numeral', 'offset', 'prediction'
]].to_json(
    '/root/NTCIR_FinNum3/ZH/Result/FinNum3_zh_test_dr.json',
    orient='records') # indent=2