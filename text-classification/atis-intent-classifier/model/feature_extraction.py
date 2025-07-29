import torch
from transformers import DistilBertTokenizer, DistilBertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "distilbert-base-uncased"  # uncased: accepts upper and lower letters

# Load once for reuse
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertModel.from_pretrained(model_path).to(device)


def feature_gen(text):
    encoded = tokenizer(text, 
                        add_special_tokens=True,
                        padding=True,               # pad to longest in batch
                        truncation=True,            # truncate to BERT's limit
                        return_tensors='pt')

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy(dtype=np.float32)

    return cls_embeddings   # (batch_size, hidden_size)
