import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

# load the model
tokenizer = GPT2Tokenizer.from_pretrained('/Users/zyw/Documents/EECS545/project/fine_tuned_gpt_model')
model = GPT2LMHeadModel.from_pretrained('/Users/zyw/Documents/EECS545/project/fine_tuned_gpt_model')
model.config.output_hidden_states = True

def gpt_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.hidden_states[-1]
    paper_embedding = torch.mean(last_hidden_state, dim=1)
    return paper_embedding