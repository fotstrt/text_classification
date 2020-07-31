from fastai.text import *
from fastai.metrics import *


import torch
import torch.nn as nn
from transformers import RobertaModel

print(torch.cuda.is_available())
from sklearn.metrics import accuracy_score

from transformers import RobertaTokenizer
roberta_tok = RobertaTokenizer.from_pretrained("roberta-base")


class FastAiRobertaTokenizer(BaseTokenizer):
    def __init__(self, tokenizer: RobertaTokenizer, max_seq_len: int=80, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    def __call__(self, *args, **kwargs):
        return self
    def tokenizer(self, t:str) -> List[str]:
        return ["<s>"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["</s>"]

fastai_tokenizer = Tokenizer(tok_func = FastAiRobertaTokenizer(roberta_tok, max_seq_len=80), pre_rules=[], post_rules=[])

path = Path()
roberta_tok.save_vocabulary(path)
with open('vocab.json', 'r') as f:
    roberta_vocab_dict = json.load(f)

fastai_roberta_vocab = Vocab(list(roberta_vocab_dict.keys()))

class RobertaTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
         super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)

class RobertaNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, vocab=fastai_roberta_vocab, **kwargs)

def get_roberta_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):
    return [RobertaTokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(vocab=vocab)]


class RobertaDataBunch(TextDataBunch):
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=80, val_bs:int=None, pad_idx=1,
               pad_first=True, device:torch.device=None, no_check:bool=False, backwards:bool=False,
               dl_tfms:Optional[Collection[Callable]]=None, **dl_kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **dl_kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs))
        return cls(*dataloaders, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

class RobertaTextList(TextList):
    _bunch = RobertaDataBunch
    _label_cls = TextList

import pandas as pd
from sklearn.model_selection import train_test_split

input_path = "../../csv_data/"
TRAIN_PROCESSED_FILE = input_path + 'train_data_full-not-processed.csv'
TEST_PROCESSED_FILE = input_path + 'test_data-not-processed.csv'

data = pd.read_csv(TRAIN_PROCESSED_FILE, header=None,  index_col=0)
data.columns=["Label", "Sentence"]
data = data.dropna()

data_train, data_val = train_test_split(data, test_size=0.1, random_state=42)
data = data_train

test_data = pd.read_csv(TEST_PROCESSED_FILE, header=None,  index_col=0)
test_data.columns=["Sentence"]

print(test_data)
processor = get_roberta_processor(tokenizer=fastai_tokenizer, vocab=fastai_roberta_vocab)
data = RobertaTextList.from_df(data, ".", cols="Sentence", processor=processor) \
    .split_by_rand_pct(valid_pct=0.00001, seed=2019) \
    .label_from_df(cols="Label",label_cls=CategoryList) \
    .add_test(RobertaTextList.from_df(test_data, ".", cols="Sentence", processor=processor)) \
    .databunch(bs=64, pad_first=False, pad_idx=0)


class CustomRobertatModel(nn.Module):
    def __init__(self,num_labels=2):
        super(CustomRobertatModel,self).__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(.05)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _ , pooled_output = self.roberta(input_ids, token_type_ids, attention_mask)
        logits = self.classifier(pooled_output)
        return logits


roberta_model = CustomRobertatModel()
predict_probabilities=False


# ----------------------------------------------------------------------------

if (predict_probabilities):

    learn=load_learner('.', file='export_base.pkl')

    x_val = data_val['Sentence'].tolist()
    y_val_true = data_val['Label'].tolist()
    y_val_pred=[]
    print(len(x_val))

    df_val=[]
    for i in range(len(x_val)):
        if (i%1000 == 0):
            print(i)
        l = learn.predict(x_val[i])
        l = l[2].detach().cpu().numpy()
        df_val.append(list(l))
        y_val_pred.append(np.argmax(l))

    print('val acc:', accuracy_score(y_val_true, y_val_pred))


    df_val = pd.DataFrame({'X': df_val})
    df_val.to_csv('../../predictions/X_val_roberta_base.csv')



    x_test = test_data['Sentence'].tolist()

    df_test=[]
    y_test_pred=[]

    for i in range(len(x_test)):
        if (i%1000 == 0):
            print(i)
        l = learn.predict(x_test[i])
        l = l[2].detach().cpu().numpy()
        df_test.append(list(l))
        y_test_pred.append(np.argmax(l))

    df_test = pd.DataFrame({'X': df_test})
    df_test.to_csv('../../predictions_new/X_test_roberta_base.csv')

# ----------------------------------------------------------------------------

learn = Learner(data, roberta_model, metrics=[accuracy])
learn.model.roberta.train() # set roberta into train mode
learn.fit_one_cycle(3, max_lr=1e-5)

learn.export('export_base.pkl')

def get_preds_as_nparray(ds_type) -> np.ndarray:

    preds = learn.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in data.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    ordered_preds = preds[reverse_sampler, :]
    pred_values = np.argmax(ordered_preds, axis=1)
    return ordered_preds, pred_values
# For Valid
preds, pred_values = get_preds_as_nparray(DatasetType.Valid)

#testing
test_preds, test_pred_values = get_preds_as_nparray(DatasetType.Test)
test_pred_values = list(test_pred_values)

print(len(test_pred_values))
idx_list = range(1, len(test_pred_values) + 1)
df = pd.DataFrame(list(zip(idx_list, test_pred_values)),
               columns =['Id', 'Prediction'])
df = df.replace(0, -1)
print(df)
df.to_csv('../../outputs/roberta_base.csv', index=False)

