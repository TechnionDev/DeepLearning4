import torch
from torch import nn
import torchtext.data
import torchtext.datasets

def load_data():
    # torchtext Field objects parse text (e.g. a review) and create a tensor representation

    # This Field object will be used for tokenizing the movie reviews text
    # For this application, tokens ~= words
    review_parser = torchtext.data.Field(
        sequential=True, use_vocab=True, lower=True,
        init_token='<sos>', eos_token='<eos>', dtype=torch.long,
        tokenize='spacy', tokenizer_language='en_core_web_sm'
    )

    # This Field object converts the text labels into numeric values (0,1,2)
    label_parser = torchtext.data.Field(
        is_target=True, sequential=False, unk_token=None, use_vocab=True
    )
    # Load SST, tokenize the samples and labels
    # ds_X are Dataset objects which will use the parsers to return tensors
    ds_train, ds_valid, ds_test = torchtext.datasets.SST.splits(
        review_parser, label_parser, root="project/data", fine_grained=True
    )
    vocab = review_parser.build_vocab(ds_train, vectors="glove.6B.300d")
    label_parser.build_vocab(ds_train)
    
    # print(f"review parser dict is {review_parser.vocab.vectors}")
    return ds_train, ds_valid, ds_test, review_parser.vocab.vectors,review_parser,label_parser, vocab
