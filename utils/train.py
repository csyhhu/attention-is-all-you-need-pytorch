import torch
import torch.nn as nn

from utils.miscellaneous import progress_bar

def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for batch_idx, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

            progress_bar(batch_idx, len(iterator), 'Testing...')

    return epoch_loss / len(iterator)