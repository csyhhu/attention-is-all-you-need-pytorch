from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, Dataset

import spacy
import numpy as np

import random
import math
import time

import dill as pickle
from transformer import Constants

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    spacy_de = spacy.load('de')
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en.tokenizer(text)]


def get_dataloader(dataset_name, batch_size, device='cpu'):

    if dataset_name == 'Multi30k':

        SRC = Field(tokenize=tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)

        TRG = Field(tokenize=tokenize_en,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)

        # SRC = Field(tokenize="spacy",
        #             tokenizer_language="de",
        #             init_token='<sos>',
        #             eos_token='<eos>',
        #             lower=True)
        #
        # TRG = Field(tokenize="spacy",
        #             tokenizer_language="en",
        #             init_token='<sos>',
        #             eos_token='<eos>',
        #             lower=True)

        train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

        SRC.build_vocab(train_data, min_freq=2)
        TRG.build_vocab(train_data, min_freq=2)

        trainloader, validloader, testloader = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=batch_size,
            device=device)

    else:
        raise NotImplementedError

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    print(f"Unique tokens in source  vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")
    return trainloader, validloader, testloader, SRC, TRG


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    #========= Preparing Model =========#
    # if opt.embs_share_weight:
    #     assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
    #         'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator


if __name__ == '__main__':

    trainloader, validloader, testloader, SRC, TRG = get_dataloader(dataset_name='Multi30k', batch_size=10)

    for batch in trainloader:
        break