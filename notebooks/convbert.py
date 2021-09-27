import tokenizers
import transformers
from transformers import (ConvBertModel, ConvBertForMaskedLM,
                          PreTrainedTokenizerFast, ConvBertTokenizerFast)
from tokenizers import Tokenizer

model_convbert = ConvBertForMaskedLM.from_pretrained('/mnt/twitchat/models/convbert-small-hf')
model_convbert_legacy = ConvBertModel.from_pretrained('YituTech/conv-bert-small')
tokenizer = Tokenizer.from_file('/media/data/Projets/twitchat-ds/models/tokenizers/all_streamers.json')
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    sep_token='<sep>',
    pad_token='<pad>',
    cls_token='<cls>',
    mask_token='<mask>'
)

sentence = "Détrompe toi jusqu'à GM ça peut servir"
tokens = fast_tokenizer.encode(sentence, return_tensors='pt')
tokens
discriminator_outputs = model_convbert(tokens)
discriminator_outputs[0].tolist()


tokenizer_legacy = ConvBertTokenizerFast.from_pretrained('YituTech/conv-bert-small')
tokens_legacy = tokenizer_legacy.encode(sentence, return_tensors='pt')
discriminator_outputs_legacy = model_convbert_legacy(tokens_legacy)

discriminator_outputs_legacy
