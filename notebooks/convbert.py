import tokenizers
import transformers
from transformers import (ConvBertModel, ConvBertForMaskedLM,
                          PreTrainedTokenizerFast, ConvBertTokenizerFast,
                          pipeline)
from tokenizers import Tokenizer

model_convbert = ConvBertForMaskedLM.from_pretrained('/mnt/twitchat/models/convbert-small-hf-mlm/checkpoint-42000')
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


fill_mask = pipeline(
    "fill-mask",
    model=model_convbert,
    tokenizer=fast_tokenizer
)

fill_mask('<mask> la fronce')
fill_mask('<mask> la france !')
fill_mask('<mention><mask>')
fill_mask('On est pas des<mask>')
fill_mask('On est pas des dau<mask>, mais des requins')
fill_mask('On est pas des<mask>, mais des requins')
fill_mask("La position est<mask>, mais ca va le faire")
fill_mask("J'aime<mask>")
fill_mask("<3<mask>")
fill_mask("slt tt le<mask>")
fill_mask("LUL LUL LUL<mask> LUL LUL LUL")
fill_mask("squeezie vs<mask>")
fill_mask("Je vais me faire<mask> !")

# Sauvegarder le tokenizer
tokenizer = Tokenizer.from_file('/mnt/twitchat/models/tokenizers/all_streamers.json')
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    sep_token='<sep>',
    pad_token='<pad>',
    cls_token='<cls>',
    mask_token='<mask>'
)
fast_tokenizer.save_pretrained('/mnt/twitchat/models/convbert-small-hf')


# SIMCSE // TRAINING
from sentence_transformers import SentenceTransformer, LoggingHandler, InputExample
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
import pandas as pd
from twitchatds import prepare_data_for_sbert

model_name = '/mnt/twitchat/models/convbert-small-hf'
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

pd_datas = prepare_data_for_sbert(pd.read_pickle('/mnt/twitchat/data/raw/zerator_squeezie_samueletienne_ponce_mistermv_jeanmassietaccropolis_domingo_blitzstream_antoinedaniellive.pkl'))
pd_datas.columns
train_sentences = pd_datas.groupby('channel').sample(20)['message_clean'].tolist()
train_sentences

train_data = [InputExample(texts=[s, s]) for s in train_sentences]

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

train_loss = losses.MultipleNegativesRankingLoss(model)

# Call the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    show_progress_bar=True
)

model.save('/mnt/twitchat/models/convbert-small-sbert-simcse/')


# SIMCSE // TEST
model = SentenceTransformer('/mnt/twitchat/models/convbert-small-simcse/')

# SIMCSE // OWN CLASSES TEST
import pandas as pd
from twitchatds import prepare_data_for_stats
from twitchatds.clustering import TwitchatSimCSEEmbedding, TwitchatUmapClustering
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file('/mnt/twitchat/models/tokenizers/all_streamers.json')
model_path_or_name = "/mnt/twitchat/models/convbert-small-simcse/"
all_streams = pd.read_pickle('/mnt/twitchat/data/raw/zerator_squeezie_samueletienne_ponce_mistermv_jeanmassietaccropolis_domingo_blitzstream_antoinedaniellive.pkl')
one_stream = all_streams[(all_streams.channel == 'blitzstream') & (all_streams.published_date == '2021-07-06')]
one_stream_stats = prepare_data_for_stats(one_stream.reset_index(), tokenizer)

one_stream_stats.head()

embedding = TwitchatSimCSEEmbedding(one_stream_stats, model_path_or_name)
embedding.compute_embeddings()
umap_clustering = TwitchatUmapClustering(embedding)
umap_clustering.compute_labels(n_components=20)

umap_clustering.describe_clusters()
umap_clustering.print_cluster(list(range(167)))
