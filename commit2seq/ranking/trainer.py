from data_manger import DataManger
from cnn import CNN
from attn_rnn import AttentionNet
from gru import GRU
from rnn import RNN
import torch.optim as optim
from tqdm import tqdm
import sys
import pandas as pd
import torch
import datetime
import argparse
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from shutil import copyfile
folder_path = os.path.dirname(os.path.abspath(__file__))


class Trainer:
    def __init__(self, args):
        self.data_manger = DataManger(args)
        self.msg_len = args.msg_len
        self.diff_len = args.diff_len
        self.arc = args.arc
        self.train_dl, self.val_dl, self.test_dl = self.data_manger.get_iterator()
        self.model = self.get_model()
        self.opt = self.get_optimizer()
        self.set_device()
        self.cur_epoch = 0
        self.best_loss = 10
        self.best_bleu = 0
        self.model_output_folder = os.path.join(folder_path, 'ranking_models')
        self.model_path = os.path.join(os.path.join(folder_path, 'ranking_models'), args.model)
        self.epoches = 500
        self.train_losses, self.valid_losses = [], []
        self.train_acc, self.valid_acc = [], []
        self.train_f1, self.valid_f1 = [], []
        self.top = args.top

    def get_model(self):
        if self.arc == 'cnn':
            print('current model is cnn')
            return CNN(diff_vectors=self.data_manger.DIFF.vocab.vectors,
                       msg_vectors=self.data_manger.RETRIEVED.vocab.vectors,
                       msg_len=self.msg_len, diff_len=self.diff_len)
        elif self.arc == 'rnn':
            print('current model is rnn')
            return RNN(diff_vectors=self.data_manger.DIFF.vocab.vectors,
                       msg_vectors=self.data_manger.RETRIEVED.vocab.vectors)
        elif self.arc == 'gru':
            print('current model is gru')
            return GRU(diff_vectors=self.data_manger.DIFF.vocab.vectors,
                       msg_vectors=self.data_manger.RETRIEVED.vocab.vectors)
        else:
            print('current model is attention')
            return AttentionNet(diff_vectors=self.data_manger.DIFF.vocab.vectors, msg_vectors=self.data_manger.RETRIEVED.vocab.vectors)

    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=1e-4)

    def start_train(self):
        print(self.data_manger.train_file)
        print(self.data_manger.val_file)
        print(self.data_manger.test_file)
        print('Train set has %d batches' % (len(self.train_dl)))
        print('Validation set has %d batches:' % (len(self.val_dl)))
        print('Test set has %d batches:' % (len(self.test_dl)))
        for i in range(self.epoches):
            self.cur_epoch = i
            self.model = self.model.train()
            train_loss_epoch = self.train_single_epoch()
            self.train_losses.append(train_loss_epoch)
            with torch.no_grad():
                self.model = self.model.eval()
                valid_loss_epoch = self.validation()
                self.valid_losses.append(valid_loss_epoch)
                self.start_test()
        print('After %d epoch, best bleu reaches: %f' % (self.epoches, self.best_bleu))
        self.plot()

    def plot(self):
        plot_folder_path = os.path.join(folder_path, 'plot')
        if not os.path.exists(plot_folder_path):
            os.makedirs(plot_folder_path)

        plt.plot(self.train_losses, color='skyblue', linewidth=2, label='train_losses')
        plt.plot(self.valid_losses, color='olive', linewidth=2, label='valid_losses')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(plot_folder_path, 'loss_fig.png'))
        plt.clf()

        plt.plot(self.train_acc, color='skyblue', linewidth=2, label='train_acc')
        plt.plot(self.valid_acc, color='olive', linewidth=2, label='valid_acc')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.savefig(os.path.join(plot_folder_path, 'acc_fig.png'))
        plt.clf()

        plt.plot(self.train_f1, color='skyblue', linewidth=2, label='train_f1')
        plt.plot(self.valid_f1, color='olive', linewidth=2, label='valid_f1')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('f1')
        plt.savefig(os.path.join(plot_folder_path, 'f1_fig.png'))
        plt.clf()

    def start_test(self):
        self.test()
        p = subprocess.Popen('python3 /home/shangqing/Documents/GitHub/c2m/evaluation/evaluate.py '
                             '-r /home/shangqing/Documents/GitHub/c2m/commit2seq/ranking/ranking_'
                             'models/raw_commit_msgs_' + str(self.top) +
                             '.txt -c /home/shangqing/Documents/GitHub/c2m/commit2seq/ranking'
                             '/ranking_models/retrieved_results_' + str(self.top) + '.txt', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        text = p.stdout.read()
        bleu_txt = text.decode('utf-8').split('BLEU = ')[1].split(',')[0]
        print('Current epoch, bleu reaches ' + bleu_txt)
        if self.best_bleu < float(bleu_txt):
            self.best_bleu = float(bleu_txt)
            copyfile('ranking_models/retrieved_results_' + str(self.top) + '.txt',
                     'ranking_models/retrieved_results_best.txt')
            copyfile('ranking_models/raw_commit_msgs_' + str(self.top) + '.txt',
                     'ranking_models/raw_commit_msgs_best.txt')

    def train_single_epoch(self, dedug=False):
        running_loss = 0
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print('Current Time: %s, Start Training at Epoch %i' % (current_time, self.cur_epoch))
        for commit_ids, commit_msgs, commit_used_tokens_tuples, retrieved_ids, retrieved_msg_tuples, rouge, bleu, \
            references in tqdm(self.train_dl, file=sys.stdout, disable=True):
            used_tokens, used_tokens_lens = commit_used_tokens_tuples
            retrieved_msgs, retrieved_msgs_lens = retrieved_msg_tuples
            if dedug:
                orig_retrieved_msgs = self.data_manger.RETRIEVED.reverse(retrieved_msgs)
                orig_retrieved_diffs = self.data_manger.DIFF.reverse(used_tokens)
                print(commit_ids, commit_msgs, retrieved_ids, orig_retrieved_diffs, orig_retrieved_msgs,
                      used_tokens_lens, retrieved_msgs_lens)
            retrieved_msgs, retrieved_msgs_len, used_tokens, used_tokens_lens = retrieved_msgs.cuda(), retrieved_msgs_lens.cuda(), used_tokens.cuda(), used_tokens_lens.cuda()
            outputs = self.model(retrieved_msgs, used_tokens)
            outputs = torch.squeeze(outputs)
            loss = self.loss_func(outputs, rouge)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            running_loss += loss
        print('Training at Epoch %d Mean Loss: %f' % (self.cur_epoch, running_loss / len(self.train_dl)))
        mean_loss = running_loss / len(self.train_dl)
        return mean_loss

    def validation(self):
        running_loss = 0
        for commit_ids, commit_msg, commit_used_tokens_tuples, retrieved_ids, retrieved_msg_tuples, rouge, bleu, \
            references in tqdm(self.val_dl, file=sys.stdout, disable=True):
            used_tokens, used_tokens_len = commit_used_tokens_tuples
            retrieved_msgs, retrieved_msgs_len = retrieved_msg_tuples
            retrieved_msgs, retrieved_msgs_len, used_tokens, used_tokens_len = retrieved_msgs.cuda(), retrieved_msgs_len.cuda(), used_tokens.cuda(), used_tokens_len.cuda()
            outputs = self.model(retrieved_msgs, used_tokens)
            outputs = torch.squeeze(outputs)
            loss = self.loss_func(outputs, rouge)
            running_loss += loss.data.cpu().tolist()
        print('Validation at Epoch %d Mean Loss: %f' % (self.cur_epoch, running_loss / len(self.val_dl)))
        if running_loss / len(self.val_dl) < self.best_loss:
            self.best_loss = running_loss / len(self.val_dl)
            self.save_models()
        return running_loss / len(self.val_dl)

    def loss_func(self, outputs, labels, custom=True):
        if custom:
            return torch.nn.SmoothL1Loss(reduction='mean')(outputs, labels).cuda()
        else:
            loss_class_weighted = torch.nn.BCEWithLogitsLoss()(outputs, labels).cuda()
            return loss_class_weighted

    def set_device(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.model.to(device)

    def compute_metrics(self, preds, labels):
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(preds, labels)
        return acc, prec, recall, f1

    def save_models(self):
        if not os.path.exists(self.model_output_folder):
            os.makedirs(self.model_output_folder)
        file_path = os.path.join(self.model_output_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%s") + '-' + str(self.top))
        torch.save(self.model.state_dict(), file_path)
        print('model save finished')

    def test(self):
        # model = self.get_model()
        # model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        running_loss = 0
        commit_ids_test, retrieved_ids_test = [], []
        orig_retrieved_msgs_test, orig_commit_msgs_test, references_test = [], [], []
        predictions_test_scores, test_scores = [], []
        for commit_ids, commit_msg, commit_used_tokens_tuples, retrieved_ids, retrieved_msg_tuples, rouge, bleu, \
            references in tqdm(self.test_dl, file=sys.stdout, disable=True):
            used_tokens, used_tokens_len = commit_used_tokens_tuples
            retrieved_msgs, retrieved_msgs_len = retrieved_msg_tuples
            orig_retrieved_msgs = self.data_manger.RETRIEVED.reverse(retrieved_msgs)
            retrieved_msgs, retrieved_msgs_len, used_tokens, used_tokens_len = retrieved_msgs.cuda(), retrieved_msgs_len.cuda(), used_tokens.cuda(), used_tokens_len.cuda()
            outputs = self.model(retrieved_msgs, used_tokens)
            outputs = torch.squeeze(outputs)
            predictions_test_scores.extend(outputs.cpu().tolist())
            loss = self.loss_func(outputs, rouge)
            running_loss += loss.data.cpu().tolist()
            commit_ids_test.extend(commit_ids)
            retrieved_ids_test.extend(retrieved_ids)
            orig_retrieved_msgs_test.extend(orig_retrieved_msgs)
            orig_commit_msgs_test.extend(commit_msg)
            references_test.extend(references)
            test_scores.extend(rouge.cpu().tolist())
        print('Test at Epoch %d Mean Loss: %f' % (self.cur_epoch, running_loss / len(self.test_dl)))
        self.write_test_results_csv(zip(commit_ids_test, orig_commit_msgs_test, retrieved_ids_test,
                                        orig_retrieved_msgs_test, references_test, predictions_test_scores, test_scores))
        self.write_test_results_text()

    def write_test_results_csv(self, ziplist):
        file_path = os.path.join(self.model_output_folder, 'model_predicted_scores' + str(self.top) + '.csv')
        df = pd.DataFrame(list(ziplist), columns=['commit_id', 'commit_msg', 'retrieved_commit_id', 'retrieved_msg',
                                                  'retrieved_reference', 'predicted_score', 'score'])
        with open(file_path, 'w') as f:
            df.to_csv(f, index=False)

    def write_test_results_text(self):
        csv_file_path = os.path.join(self.model_output_folder, 'model_predicted_scores' + str(self.top) + '.csv')
        df = pd.read_csv(csv_file_path)
        scores = df['predicted_score'].tolist()
        retrieved_msgs = df['retrieved_reference'].tolist()
        commit_msgs = df['commit_msg'].tolist()
        trunk_length = int(self.top) + 1
        trunk_nums = int(len(scores) / trunk_length)
        commit_msg_results = []
        retrieved_msg_results = []
        for i in range(trunk_nums):
            trunk_scores = scores[i * trunk_length: (i + 1) * trunk_length]
            trunk_retrieved_msgs = retrieved_msgs[i * trunk_length: (i + 1) * trunk_length]
            trunk_commit_msgs = commit_msgs[i * trunk_length: (i + 1) * trunk_length]
            max_index = np.argmax(trunk_scores)
            retrieved_msg_results.append(trunk_retrieved_msgs[max_index])
            commit_msg_results.append(trunk_commit_msgs[max_index])
        with open(os.path.join(self.model_output_folder, 'retrieved_results_' + str(self.top) + '.txt'), 'w') as f:
            f.write('\n'.join(retrieved_msg_results))
        with open(os.path.join(self.model_output_folder, 'raw_commit_msgs_' + str(self.top) + '.txt'), 'w') as f:
            f.write('\n'.join(commit_msg_results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ranking')
    parser.add_argument('-f', '--folder', default=os.path.join(os.path.dirname(folder_path), 'training_data/ranking/'))
    parser.add_argument('-m', '--msg_len', default=15)
    parser.add_argument('-d', '--diff_len', default=200)
    parser.add_argument('-t', '--type', default='train')
    parser.add_argument('-p', '--model', default='20191019-08091571443788')
    parser.add_argument('-top', '--top', default=1)
    parser.add_argument('-a', '--arc', default='CNN')
    args = parser.parse_args()
    trainer = Trainer(args)
    if args.type == 'train':
        trainer.start_train()
    else:
        print('test begin')
        trainer.start_test()
