import torch
import torch.nn as nn
import torch.optim as optim

from utils.miscellaneous import progress_bar
from train import prepare_dataloaders
from seq2seq.model import Seq2Seq, Attention, Encoder, Decoder

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data_pkl', default='./m30k_deen_shr.pkl')     # all-in-1 data pickle or bpe field

parser.add_argument('-train_path', default=None)   # bpe encoded data
parser.add_argument('-val_path', default=None)     # bpe encoded data

parser.add_argument('-epoch', '-epoch', type=int, default=10)
parser.add_argument('-b', '--batch_size', type=int, default=2048)

parser.add_argument('-d_model', type=int, default=512)
parser.add_argument('-d_inner_hid', type=int, default=2048)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)

parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-n_layers', type=int, default=6)
parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)

parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-embs_share_weight', action='store_true')
parser.add_argument('-proj_share_weight', action='store_true')

parser.add_argument('-log', default=None)
parser.add_argument('-save_model', default=None)
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-label_smoothing', action='store_true')

opt = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -------------------

# ------------
# Load Dataset
# ------------
training_data, validation_data = prepare_dataloaders(opt, device)

# ----------------
# Initialize Model
# ----------------
INPUT_DIM = opt.src_vocab_size
OUTPUT_DIM = opt.trg_vocab_size
ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=opt.trg_pad_idx)

optimizer = optim.Adam(model.parameters())

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# --------------
# Begin Training
# --------------
clip = 1
for epoch_idx in range(opt.epoch):

    model.train()

    train_loss = 0
    for batch_idx, batch in enumerate(training_data):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        losses = criterion(output, trg)

        losses.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        # ------
        # Record
        # ------
        train_loss += losses.item()

        progress_bar(batch_idx, len(training_data), train_loss / (batch_idx + 1))
