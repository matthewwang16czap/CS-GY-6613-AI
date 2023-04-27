from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import streamlit as st

# load tokenizer and fine-tuned model
save_diirectory = "saved"
model = DistilBertForSequenceClassification.from_pretrained(save_diirectory)
model.eval()
tokenizer = DistilBertTokenizerFast.from_pretrained(save_diirectory)

# method to transform dataset to (patent_numbers, abstracts, claims, texts, labels), labels are 1/0 from decision
def dataset_to_lists(dataset):
    patent_numbers = []
    abstracts = []
    claims = []
    texts = []
    labels = []
    for data in dataset:
        patent_number = data['patent_number']
        abstract = data['abstract']
        claim = data['claims']
        text = data['abstract'] + data['claims']
        label = 1 if data['decision'] == 'ACCEPTED' else 0
        patent_numbers.append(patent_number)
        abstracts.append(abstract)
        claims.append(claim)
        texts.append(text)
        labels.append(label)
    return patent_numbers, abstracts, claims, texts, labels

# dataset class to fit in dataloader
class TextEncodeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


# load dataset, only focus on "patent_number", "abstract", "claims", "decision" at Jan
dataset_dict = load_dataset('HUPD/hupd',
    name='sample',
    data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather", 
    icpr_label=None,
    train_filing_start_date='2016-01-01',
    train_filing_end_date='2016-01-31',
    val_filing_start_date='2016-02-01',
    val_filing_end_date='2016-02-01',
)
dataset = dataset_dict['train']
dataset.set_format(type="torch", columns=["patent_number", "abstract", "claims", "decision"])

# transform dataset to lists of data
patent_numbers, abstracts, claims, texts, labels = dataset_to_lists(dataset)


# select a patent_number and get relevant info
patent_number = str(st.selectbox('select a patent', patent_numbers))
selected_idx = patent_numbers.index(patent_number)
abstract = abstracts[selected_idx]
claim = claims[selected_idx]
text = texts[selected_idx]
label = labels[selected_idx]

# display abstract and claim
st.write('abstract: ' + abstract)
st.write('claim: ' + claim)

# get encoding text
encoding = tokenizer(text, truncation=True, padding=True)

# click to make prediction, the score is the probability of accepted
if st.button('Run'):
    with torch.no_grad():
        output = model(torch.tensor(encoding['input_ids']).unsqueeze(dim=0), attention_mask=torch.tensor(encoding['attention_mask']).unsqueeze(dim=0), labels=torch.tensor(label))
        predictions = F.softmax(output.logits, dim=1)
        score = predictions[0][1].item()
        st.write('score: ' + str(score))
        st.write('actual result: ' + str(label))
        st.write('output: ' + str(output))
    