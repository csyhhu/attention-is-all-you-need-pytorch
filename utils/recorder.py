"""
A class to incorprate all necessary function to record training log
"""
import time
import os
import shutil

import numpy as np

import torch

from utils.miscellaneous import  progress_bar, AverageMeter

class Recorder():
    """
    A class to record training log and write into txt file
    """

    def __init__(self, SummaryPath):

        self.SummaryPath = SummaryPath

        if not os.path.exists(SummaryPath):
            os.makedirs(SummaryPath)
        else:
            if os.path.exists('%s/ckpt.pt' %SummaryPath):
                action = input('r for resume training, d for delete the previous log')
                if action == 'd':
                    print('Record exist, remove')
                    shutil.rmtree(SummaryPath)
                    os.makedirs(SummaryPath)
            else:
                print('Record exist, remove')
                shutil.rmtree(SummaryPath)
                os.makedirs(SummaryPath)

        print('Summary records saved at: %s' %SummaryPath)

        self.loss = AverageMeter()
        self.niter = 0
        self.stop = False

        ###################
        # Initialize file #
        ###################
        self.train_loss_record = open('%s/train-loss.txt' % (self.SummaryPath), 'a+')
        # self.train_perplexity_record = open('%s/perplexity.txt' % (self.SummaryPath), 'a+')
        self.test_loss_record = open('%s/test-loss.txt' % (self.SummaryPath), 'a+')
        self.lr_record = open('%s/lr.txt' % (self.SummaryPath), 'a+')
        self.args_record = open('%s/arguments.txt' % (self.SummaryPath), 'w+')


    def write_arguments(self, args_list):

        for args in args_list:
            if isinstance(args, dict):
                for key, value in args.items():
                    self.args_record.write('%s: %s\n' %(key, value))
            else:
                for arg in vars(args):
                    self.args_record.write('%s: %s\n' %(arg, getattr(args, arg)))

            self.flush(self.args_record)

        self.args_record.close()


    def update(self, cur_loss=0, batch_size=0, cur_lr=1e-3, is_train = True):

        if is_train:
            self.niter += 1

            self.loss.update(cur_loss, batch_size)
            self.train_loss_record.write(
                '%d, %.8f, %.8f\n' % (self.niter, self.loss.val, self.loss.avg)
            )
            self.lr_record.write(
                '%d, %e' %(self.niter, cur_lr)
            )
            # self.perplexity_record.write(
            #
            # )

            self.flush([self.train_loss_record, self.lr_record])

        else:
            self.test_loss_record.write('%d, %.8f\n' % (self.niter, cur_loss))
            self.flush([self.test_loss_record])


    def restart_training(self):
        self.reset_performance()
        self.stop = False


    def reset_performance(self):

        self.loss.reset()


    def flush(self, file_list=None):
        for file in file_list:
            file.flush()


    def close(self):
        self.train_loss_record.close()
        self.test_loss_record.close()


    def print_training_result(self, batch_idx, n_batch, append=None):

        progress_bar(batch_idx, n_batch, "Loss: %.3f (%.3f) %s" % (self.loss.val, self.loss.avg, None if append is None else "| %s" % append))


if __name__ == '__main__':
    pass





