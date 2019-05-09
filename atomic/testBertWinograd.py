import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from IPython import embed

def loadBERT():
  global tokenizer
  global model
  print("Loading BERT")
  # Load pre-trained model tokenizer (vocabulary)
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  # Load pre-trained model (weights)
  model = BertForMaskedLM.from_pretrained('bert-base-uncased')
  model.eval()
  print("Done")

def smallSet():
  texts = ["Olivia takes Taylor's advice, before _ makes a decision.",
           "Taylor takes Olivia's advice, before _ makes a decision.",
           "Jesse takes Kai's advice, before _ makes a decision.",
           "Jennifer takes Bonnie's advice, before _ saw her the next time.",
           "Toni takes Lauren's advice, before _ goes to report to police.",
           "Marissa takes Brittany's advice, before _ decides to listen to someone else."]
  answers = [["Olivia","Taylor"],["Taylor","Olivia"],["Jesse","Kai"],
             ["Bonnie","Jennifer"],["Lauren","Toni"],["Marissa","Brittany"]]
  assert len(texts) == len(answers)
  for t,a in zip(texts,answers):
    evalSentences(t,a)
  
def evalSentences(text,answers):
  # Tokenized input
  tokenized_text = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
  answers = [a.lower() for a in answers]
  
  # Mask a token that we will try to predict back with `BertForMaskedLM`
  masked_index = tokenized_text.index("_")
  tokenized_text[masked_index] = '[MASK]'

  # assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']

  # Convert token to vocabulary indices

  # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
  # segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  segment_boundary = tokenized_text.index(",")
  tokenized_text.insert(segment_boundary,"[SEP]")
  segments_ids = [0 if i <= segment_boundary -1 else 1 for i,_ in enumerate(tokenized_text)]
  # print(list(zip(tokenized_text,segments_ids)))
  # Convert inputs to PyTorch tensors
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])


  # Predict all tokens
  predictions = model(tokens_tensor, segments_tensors)
  answer_ix = tokenizer.convert_tokens_to_ids(answers)

  # confirm we were able to predict answer
  # predicted_index = torch.argmax(predictions[0, masked_index]).item()
  predicted_index = torch.softmax(predictions[0, masked_index],dim=0)
  print(text)
  embed();exit()
  for i,(a,ix) in enumerate(zip(answers,answer_ix)):
    p = predicted_index[ix].item()
    print(f"{a}: {p:.4g}"+( " <-" if i == 0 else ""))
  
if __name__ == "__main__":
  loadBERT()
  smallSet()


  # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
  # assert predicted_token == 'henson'
