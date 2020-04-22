import torch
import torch.nn as nn
import torch.optim as optim

from utils.miscellaneous import progress_bar, count_parameters
from utils.recorder import Recorder
from utils.dataset import get_dataloader, prepare_dataloaders
from utils.train import evaluate

from seq2seq.RecurrentSeq2Seq import RecSeq2Seq
from seq2seq.AttenSeq2Seq import AttenSeq2Seq
from seq2seq.Transformer import Transformer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', '-model', type=str, default='rec')
parser.add_argument('--dataset_name', '-dataset', type=str, default='Multi30k')
parser.add_argument('--data_pkl', '-data_pkl', default='m30k_deen_shr.pkl')
parser.add_argument('--batch_size', '-b', type=int, default=128)
# ------------------
# Model Arch Arguments
# ------------------
parser.add_argument('-enc_dim', type=int, default=256)
parser.add_argument('-dec_dim', type=int, default=256)
parser.add_argument('-hidden_size', type=int, default=512)
parser.add_argument('-enc_dropout', type=float, default=0.5)
parser.add_argument('-dec_dropout', type=float, default=0.5)
# ------------------
# Training Arguments
# ------------------
parser.add_argument('-clip', type=float, default=1.0)
parser.add_argument('-epoch', type=int, default=10)
# parser.add_argument('-embs_share_weight', action='store_true')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -------------------

# ------------
# Load Dataset
# ------------
# train_loader, eval_loader, test_loader, SRC, TRG = \
#     get_dataloader(dataset_name=args.dataset_name, batch_size=args.batch_size, device=device)
train_loader, test_loader = prepare_dataloaders(args, device)

src_vocab_size = args.src_vocab_size
trg_vocab_size = args.trg_vocab_size
src_ignore_index = args.src_pad_idx
trg_ignore_index = args.trg_pad_idx

# ----------------
# Initialize Model
# ----------------
if args.model_name == 'rec':
    model = RecSeq2Seq(
        input_dim=src_vocab_size, output_dim=trg_vocab_size,
        enc_emb_dim=args.enc_emb_dim, dec_emb_dim=args.dec_emb_dim,
        enc_hidden_dim=args.enc_hidden_dim, dec_hidden_dim=args.dec_hidden_dim,
        enc_dropout=args.enc_dropout, dec_dropout=args.dec_dropout, device=device
    )
elif args.model_name == 'atten':
    model = AttenSeq2Seq(
        input_dim=src_vocab_size, output_dim=trg_vocab_size,
        enc_emb_dim=args.enc_emb_dim, dec_emb_dim=args.dec_emb_dim,
        enc_hidden_dim=args.enc_hidden_dim, dec_hidden_dim=args.dec_hidden_dim,
        enc_dropout=args.enc_dropout, dec_dropout=args.dec_dropout, device=device
    )
elif args.model_name == 'transformer':
    model = Transformer(
        input_dim=src_vocab_size, output_dim=trg_vocab_size, src_pad_idx=src_ignore_index, trg_pad_idx=trg_ignore_index,
        hidden_dim=args.enc_hidden_dim,
        enc_dropout=args.enc_dropout, dec_dropout=args.dec_dropout, device=device
    )
else:
    raise NotImplementedError

print(f'The model has {count_parameters(model):,} trainable parameters')

# -----------
# Initialize Criterion and Optimizer
# -----------
criterion = nn.CrossEntropyLoss(ignore_index=trg_ignore_index)
optimizer = optim.Adam(model.parameters())

# -----
# Initialize Recorder
# -----
SummaryPath = './Reults/Seq2Seq/%s/runs' %(args.model_name)
recoder = Recorder(SummaryPath)
if recoder is not None:
    recoder.write_arguments([args])

# --------------
# Begin Training
# --------------
for epoch_idx in range(args.epoch):

    model.train()

    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        losses = criterion(output, trg)

        losses.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        # ------
        # Record
        # ------
        if recoder is not None:
            recoder.update(losses.item(), batch_size=args.batch_size, cur_lr=optimizer.param_groups[0]['lr'])
            recoder.print_training_result(batch_idx, len(train_loader))
        else:
            train_loss += losses.item()
            progress_bar(batch_idx, len(train_loader), "Loss: %.3f" %(train_loss / (batch_idx + 1)))

    # -----
    # Test
    # -----
    eval_loss = evaluate(model, test_loader, criterion)
    if recoder is not None:
        recoder.update(eval_loss, is_train=False)
    print('[%2d] Test loss: %.3f' % (epoch_idx, eval_loss))

if recoder is not None:
    recoder.close()