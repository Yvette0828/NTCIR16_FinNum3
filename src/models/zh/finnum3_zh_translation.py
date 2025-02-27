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


trn_json_f = open('/root/FinNum-3_train.json')
trn_json = json.load(trn_json_f)
trn_json_f.close()

trn_df = pd.read_json('/root/FinNum-3_train.json')


trn_df['translated_text'] = ''
trn_df['index'] = ''

trn_df.head()

translator_zh_en = pipeline("translation", model = 'Helsinki-NLP/opus-mt-zh-en')
translator_en_zh = pipeline("translation", model = 'Helsinki-NLP/opus-mt-en-zh')

def translation_aug(text):
  translation_aug = translator_en_zh(translator_zh_en(text)[0]['translation_text'])
  translation_aug = translation_aug[0]['translation_text']
  return translation_aug

for i in range(len(trn_df)):
  trn_df.loc[i, 'translated_text'] = translation_aug(trn_df.loc[i, 'text'])
  # find the index of target_numeral
  text = trn_df.loc[i, 'translated_text']
  if str(int(trn_df.loc[i, 'target_numeral'])) in trn_df.loc[i, 'translated_text']:
    index = text.index(str(int(trn_df.loc[i, 'target_numeral'])))
    trn_df.loc[i, 'index'] = index
  else:
    trn_df.loc[i, 'index'] = trn_df.loc[i, 'offset']

trn_df.to_csv('/root/FinNum-3_translation.csv')
# trn_df.to_csv('/root/FinNum-3_translation_utf-8-sig.csv', encoding = 'utf-8-sig')

###
trn = pd.read_csv('/root/FinNum-3_translation.csv')
trn_df = trn[['translated_text', 'target_numeral', 'category', 'index', 'claim']]
trn_df.columns = ['text', 'target_numeral', 'category', 'offset', 'claim']
print(trn_df.head())

trn_df.to_json('/root/FinNum-3_translated_trn.json')