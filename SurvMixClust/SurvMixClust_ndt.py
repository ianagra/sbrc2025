import numpy as np
import pandas as pd

# Criar dataset de survival



df = pd.read_pickle('../dataset/survival_ndt.pkl')
with open('../dataset/survival_ndt.pkl', 'rb') as f:
    dict_dataset_name_2_strats = pickle.load(f)
with open('./datasets/dict_list_names.pkl', 'rb') as f:
    dict_list_names = pickle.load(f)