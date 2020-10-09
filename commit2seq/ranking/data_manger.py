from torchtext.data import TabularDataset
from torchtext.data import Field, RawField, LabelField, ReversibleField
from torchtext.data import Iterator, BucketIterator
import torch
import torchtext.vocab as vocab


class DataManger:
    def __init__(self, args, batch_first=True):
        self.path_folder = args.folder
        self.batch_size = 256
        self.train_file = 'train_retrieved_top_' + str(args.top) + '.csv'
        self.val_file = 'val_retrieved_top_' + str(args.top) + '.csv'
        self.test_file = 'test_retrieved_top_' + str(args.top) + '.csv'
        self.msg_embedding_file = 'commits_dim_256_msg_vocab.txt'
        self.msg_embedding = self.get_vector(self.msg_embedding_file, self.path_folder)
        self.diff_embedding_file = 'commits_dim_256_diff_vocab.txt'
        self.diff_embedding = self.get_vector(self.diff_embedding_file, self.path_folder)
        self.tokenize = lambda x: x.split(' ')
        self.msg_len = args.msg_len
        self.diff_len = args.diff_len
        self.type = args.type
        self.DIFF = ReversibleField(sequential=True, tokenize=self.tokenize, lower=True, fix_length=self.diff_len,
                                    include_lengths=True, batch_first=batch_first)
        self.RETRIEVED = ReversibleField(sequential=True, tokenize=self.tokenize, lower=True, fix_length=self.msg_len,
                                         include_lengths=True, batch_first=batch_first)
        self.ROUGE = Field(sequential=False, use_vocab=False, batch_first=batch_first, dtype=torch.float)
        self.BLEU = Field(sequential=False, use_vocab=False, batch_first=batch_first, dtype=torch.float)
        self.MSG = RawField()
        self.COMMIT_ID = RawField()
        self.RETRIEVED_ID = RawField()
        self.REFERENCE = RawField()
        self.train_iter, self.val_iter, self.test_iter = None, None, None
        self.get_data()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_data(self):
        csv_datafields = [('commit_id', self.COMMIT_ID), ('commit_msg', self.MSG), ('predicted_msg', None),
                         ('commit_diff', None), ('commit_used_tokens', self.DIFF), ('retrieved_id', self.RETRIEVED_ID),
                         ('retrieved_msg', self.RETRIEVED), ('retrieved_msg_ref', self.REFERENCE),
                          ('retrieved_diff', None), ('rouge-L', self.ROUGE), ('bleu4', self.BLEU)]

        self.train_set, self.val_set, self.test_set = TabularDataset.splits(self.path_folder, train=self.train_file,
                                                                            validation=self.val_file,  test=self.test_file,
                                                                            format='csv', skip_header=True, fields=csv_datafields)
        self.DIFF.build_vocab(self.train_set, self.val_set, self.test_set, vectors=self.diff_embedding)
        self.RETRIEVED.build_vocab(self.train_set, self.val_set, self.test_set, vectors=self.msg_embedding)

    def get_iterator(self):
        self.train_iter, self.val_iter = BucketIterator.splits(
            (self.train_set, self.val_set), batch_sizes=(self.batch_size, self.batch_size),
            device=self.device, sort_key=lambda x: len(x.retrieved_msg),
            sort_within_batch=False, repeat=False, shuffle=True)
        train_dl = BatchWrapper(self.train_iter, commit_id='commit_id', commit_msg='commit_msg',
                                commit_used_tokens='commit_used_tokens', retrieved_id='retrieved_id',
                                retrieved_msg='retrieved_msg', rouge='rouge-L', bleu='bleu4', reference='retrieved_msg_ref')
        val_dl = BatchWrapper(self.val_iter, commit_id='commit_id', commit_msg='commit_msg',
                              commit_used_tokens='commit_used_tokens', retrieved_id='retrieved_id',
                              retrieved_msg='retrieved_msg', rouge='rouge-L', bleu='bleu4', reference='retrieved_msg_ref')
        self.test_iter = Iterator(self.test_set, batch_size=self.batch_size, shuffle=False,
                                  device=self.device, sort=False, sort_within_batch=False, repeat=False)
        test_dl = BatchWrapper(self.test_iter, commit_id='commit_id', commit_msg='commit_msg',
                               commit_used_tokens='commit_used_tokens', retrieved_id='retrieved_id',
                               retrieved_msg='retrieved_msg', rouge='rouge-L', bleu='bleu4', reference='retrieved_msg_ref')
        print('data load successful')
        return train_dl, val_dl, test_dl

    def get_vector(self, embedding_file, path_folder):
        return vocab.Vectors(embedding_file, cache=path_folder)


class BatchWrapper:
    def __init__(self, dl, commit_id, commit_msg, commit_used_tokens, retrieved_id, retrieved_msg, rouge, bleu, reference):
        self.dl, self.commit_id, self.commit_msg, self.commit_used_tokens, self.retrieved_id, self.retrieved_msg, \
        self.rouge, self.bleu, self.reference = dl, commit_id, commit_msg, commit_used_tokens, retrieved_id, retrieved_msg,\
                                                rouge, bleu, reference

    def __iter__(self):
        for batch in self.dl:
            commit_id = getattr(batch, self.commit_id)
            commit_msg = getattr(batch, self.commit_msg)
            commit_used_tokens = getattr(batch, self.commit_used_tokens)
            retrieved_id = getattr(batch, self.retrieved_id)
            retrieved_msg = getattr(batch, self.retrieved_msg)
            reference = getattr(batch, self.reference)
            rouge = getattr(batch, self.rouge)
            bleu = getattr(batch, self.bleu)
            yield (commit_id, commit_msg, commit_used_tokens, retrieved_id, retrieved_msg, rouge, bleu, reference)

    def __len__(self):
        return len(self.dl)


