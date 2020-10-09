import os

import tensorflow as tf
from common import Common

COMMIT_ID = 'COMMIT_ID'

TARGET_INDEX_KEY = 'TARGET_INDEX_KEY'
TARGET_STRING_KEY = 'TARGET_STRING_KEY'
TARGET_LENGTH_KEY = 'TARGET_LENGTH_KEY'

POSITIVE_PATH_SOURCE_INDICES_KEY = 'POSITIVE_PATH_SOURCE_INDICES_KEY'
POSITIVE_NODE_INDICES_KEY = 'POSITIVE_NODE_INDICES_KEY'
POSITIVE_PATH_TARGET_INDICES_KEY = 'POSITIVE_PATH_TARGET_INDICES_KEY'
POSITIVE_PATH_SOURCE_LENGTHS_KEY = 'POSITIVE_PATH_SOURCE_LENGTHS_KEY'
POSITIVE_PATH_LENGTHS_KEY = 'POSITIVE_PATH_LENGTHS_KEY'
POSITIVE_PATH_TARGET_LENGTHS_KEY = 'POSITIVE_PATH_TARGET_LENGTHS_KEY'
POSITIVE_PATH_SOURCE_STRINGS_KEY = 'POSITIVE_PATH_SOURCE_STRINGS_KEY'
POSITIVE_PATH_STRINGS_KEY = 'POSITIVE_PATH_STRINGS_KEY'
POSITIVE_PATH_TARGET_STRINGS_KEY = 'POSITIVE_PATH_TARGET_STRINGS_KEY'

NEGATIVE_PATH_SOURCE_INDICES_KEY = 'NEGATIVE_PATH_SOURCE_INDICES_KEY'
NEGATIVE_NODE_INDICES_KEY = 'NEGATIVE_NODE_INDICES_KEY'
NEGATIVE_PATH_TARGET_INDICES_KEY = 'NEGATIVE_PATH_TARGET_INDICES_KEY'
NEGATIVE_PATH_SOURCE_LENGTHS_KEY = 'NEGATIVE_PATH_SOURCE_LENGTHS_KEY'
NEGATIVE_PATH_LENGTHS_KEY = 'NEGATIVE_PATH_LENGTHS_KEY'
NEGATIVE_PATH_TARGET_LENGTHS_KEY = 'NEGATIVE_PATH_TARGET_LENGTHS_KEY'
NEGATIVE_PATH_SOURCE_STRINGS_KEY = 'NEGATIVE_PATH_SOURCE_STRINGS_KEY'
NEGATIVE_PATH_STRINGS_KEY = 'NEGATIVE_PATH_STRINGS_KEY'
NEGATIVE_PATH_TARGET_STRINGS_KEY = 'NEGATIVE_PATH_TARGET_STRINGS_KEY'

POSITIVE_VALID_CONTEXT_MASK_KEY = 'POSITIVE_VALID_CONTEXT_MASK_KEY'
NEGATIVE_VALID_CONTEXT_MASK_KEY = 'NEGATIVE_VALID_CONTEXT_MASK_KEY'

NEGATIVE_DISMATCHED_INDEX_KEY = 'NEGATIVE_DISMATCHED_INDEX_KEY'
POSITIVE_DISMATCHED_INDEX_KEY = 'POSITIVE_DISMATCHED_INDEX_KEY'
NEGATIVE_DISMATCHED_LENGTH_KEY = 'NEGATIVE_DISMATCHED_LENGTH_KEY'
POSITIVE_DISMATCHED_LENGTH_KEY = 'POSITIVE_DISMATCHED_LENGTH_KEY'
POSITIVE_DISMATCHED_MASK_KEY = 'POSITIVE_DISMATCHED_MASK_KEY'
NEGATIVE_DISMATCHED_MASK_KEY = 'NEGATIVE_DISMATCHED_MASK_KEY'


class Reader:
    class_subtoken_table = None
    class_target_table = None
    class_node_table = None

    def __init__(self, subtoken_to_index, target_to_index, node_to_index, config, is_evaluating=False):
        self.config = config
        self.file_path = config.TEST_PATH if is_evaluating else (config.TRAIN_PATH + '.train.c2s')
        if self.file_path is not None and not os.path.exists(self.file_path):
            print('%s cannot find file: %s' % ('Evaluation reader' if is_evaluating else 'Train reader', self.file_path))
        self.batch_size = config.TEST_BATCH_SIZE if is_evaluating else config.BATCH_SIZE
        self.is_evaluating = is_evaluating

        self.context_pad = '{},{},{}'.format(Common.PAD, Common.PAD, Common.PAD)
        self.subtoken_table = Reader.get_subtoken_table(subtoken_to_index)
        self.target_table = Reader.get_target_table(target_to_index)
        self.node_table = Reader.get_node_table(node_to_index)
        if self.file_path is not None:
            self.output_tensors = self.compute_output()

    @classmethod
    def get_subtoken_table(cls, subtoken_to_index):
        if cls.class_subtoken_table is None:
            cls.class_subtoken_table = cls.initialize_hash_map(subtoken_to_index, subtoken_to_index[Common.UNK])
        return cls.class_subtoken_table

    @classmethod
    def get_target_table(cls, target_to_index):
        if cls.class_target_table is None:
            cls.class_target_table = cls.initialize_hash_map(target_to_index, target_to_index[Common.UNK])
        return cls.class_target_table

    @classmethod
    def get_node_table(cls, node_to_index):
        if cls.class_node_table is None:
            cls.class_node_table = cls.initialize_hash_map(node_to_index, node_to_index[Common.UNK])
        return cls.class_node_table

    @classmethod
    def initialize_hash_map(cls, word_to_index, default_value):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(word_to_index.keys()), list(word_to_index.values()),
                                                        key_dtype=tf.string,
                                                        value_dtype=tf.int32), default_value)

    def process_from_placeholder(self, row):
        parts = tf.io.decode_csv(row, field_delim=',', use_quote_delim=False)
        return self.process_dataset(*parts)

    def process_dataset(self, *row_parts):
        row_parts = list(row_parts)
        commit_id = row_parts[0]

        word, target_word_labels, clipped_target_lengths = self.process_word(row_parts[1])

        negative_dense_split_contexts = self.process_contexts(row_parts[2])    # (max_context, 3)
        positive_dense_split_contexts = self.process_contexts(row_parts[3])    # (max_context, 3)

        # tf.Print(tf.shape(negative_dense_split_contexts), [tf.shape(negative_dense_split_contexts)], message='negative_dense_split_contexts')
        # tf.Print(tf.shape(positive_dense_split_contexts), [tf.shape(positive_dense_split_contexts)], message='positive_dense_split_contexts')

        positive_path_source_strings, positive_path_source_indices, positive_path_source_lengths = self.process_context_source_string(positive_dense_split_contexts)

        positive_path_strings, positive_node_indices, positive_path_lengths = self.process_context_path_string(positive_dense_split_contexts)

        positive_path_target_strings, positive_path_target_indices, positive_path_target_lengths = self.process_context_target_string(positive_dense_split_contexts)

        negative_path_source_strings, negative_path_source_indices, negative_path_source_lengths = self.process_context_source_string(negative_dense_split_contexts)

        negative_path_strings, negative_node_indices, negative_path_lengths = self.process_context_path_string(negative_dense_split_contexts)

        negative_path_target_strings, negative_path_target_indices, negative_path_target_lengths = self.process_context_target_string(negative_dense_split_contexts)

        negative_dismatched_words, negative_dismatched_indices, negative_dismatched_lengths = self.process_dismatched(row_parts[4])
        positive_dismatched_words, positive_dismatched_indices, positive_dismatched_lengths = self.process_dismatched(row_parts[5])

        positive_valid_contexts_mask = tf.to_float(tf.not_equal(
            tf.reduce_max(positive_path_source_indices, -1) + tf.reduce_max(positive_node_indices, -1) + tf.reduce_max(positive_path_target_indices, -1), 0))          # (max_contexts)

        # tf.Print(tf.shape(positive_valid_contexts_mask),[tf.shape(positive_valid_contexts_mask)], message='positive_valid_contexts_mask')

        negative_valid_contexts_mask = tf.to_float(tf.not_equal(
            tf.reduce_max(negative_path_source_indices, -1) + tf.reduce_max(negative_node_indices, -1) + tf.reduce_max(negative_path_target_indices, -1), 0))          # (max_contexts)

        positive_dismatched_mask = tf.reshape(tf.to_float(tf.not_equal(tf.reduce_max(positive_dismatched_indices, -1), 0)), [1])
        negative_dismatched_mask = tf.reshape(tf.to_float(tf.not_equal(tf.reduce_max(negative_dismatched_indices, -1), 0)), [1])

        return {
                COMMIT_ID: commit_id,
                TARGET_STRING_KEY: word,
                TARGET_INDEX_KEY: target_word_labels,
                TARGET_LENGTH_KEY: clipped_target_lengths,

                POSITIVE_PATH_SOURCE_INDICES_KEY: positive_path_source_indices,
                POSITIVE_NODE_INDICES_KEY: positive_node_indices,
                POSITIVE_PATH_TARGET_INDICES_KEY: positive_path_target_indices,
                POSITIVE_VALID_CONTEXT_MASK_KEY: positive_valid_contexts_mask,
                POSITIVE_PATH_SOURCE_LENGTHS_KEY: positive_path_source_lengths,
                POSITIVE_PATH_LENGTHS_KEY: positive_path_lengths,
                POSITIVE_PATH_TARGET_LENGTHS_KEY: positive_path_target_lengths,
                POSITIVE_PATH_SOURCE_STRINGS_KEY: positive_path_source_strings,
                POSITIVE_PATH_STRINGS_KEY: positive_path_strings,
                POSITIVE_PATH_TARGET_STRINGS_KEY: positive_path_target_strings,

                NEGATIVE_PATH_SOURCE_INDICES_KEY: negative_path_source_indices,
                NEGATIVE_NODE_INDICES_KEY: negative_node_indices,
                NEGATIVE_PATH_TARGET_INDICES_KEY: negative_path_target_indices,
                NEGATIVE_VALID_CONTEXT_MASK_KEY: negative_valid_contexts_mask,
                NEGATIVE_PATH_SOURCE_LENGTHS_KEY: negative_path_source_lengths,
                NEGATIVE_PATH_LENGTHS_KEY: negative_path_lengths,
                NEGATIVE_PATH_TARGET_LENGTHS_KEY: negative_path_target_lengths,
                NEGATIVE_PATH_SOURCE_STRINGS_KEY: negative_path_source_strings,
                NEGATIVE_PATH_STRINGS_KEY: negative_path_strings,
                NEGATIVE_PATH_TARGET_STRINGS_KEY: negative_path_target_strings,

                NEGATIVE_DISMATCHED_INDEX_KEY: negative_dismatched_indices,
                POSITIVE_DISMATCHED_INDEX_KEY:  positive_dismatched_indices,
                NEGATIVE_DISMATCHED_LENGTH_KEY: negative_dismatched_lengths,
                POSITIVE_DISMATCHED_LENGTH_KEY: positive_dismatched_lengths,
                POSITIVE_DISMATCHED_MASK_KEY: positive_dismatched_mask,
                NEGATIVE_DISMATCHED_MASK_KEY: negative_dismatched_mask
                }

    def process_word(self, word):
        split_target_labels = tf.string_split(tf.expand_dims(word, -1), delimiter='|')
        target_dense_shape = [1, tf.maximum(tf.to_int64(self.config.MAX_TARGET_PARTS),         # (1,MAX_TARGET_PARTS)
                                            split_target_labels.dense_shape[1] + 1)]
        sparse_target_labels = tf.sparse.SparseTensor(indices=split_target_labels.indices,
                                                      values=split_target_labels.values,
                                                      dense_shape=target_dense_shape)
        dense_target_label = tf.reshape(tf.sparse.to_dense(sp_input=sparse_target_labels,
                                                           default_value=Common.PAD), [-1])    # (MAX_TARGET_PARTS,1)
        index_of_blank = tf.where(tf.equal(dense_target_label, Common.PAD))
        target_length = tf.reduce_min(index_of_blank)
        dense_target_label = dense_target_label[:self.config.MAX_TARGET_PARTS]                 # (MAX_TARGET_PARTS,1)
        clipped_target_lengths = tf.clip_by_value(target_length, clip_value_min=0,
                                                  clip_value_max=self.config.MAX_TARGET_PARTS)
        target_word_labels = tf.concat([
            self.target_table.lookup(dense_target_label), [0]],
            axis=-1)                                                                    # (max_target_parts + 1) of int 0 means padding
        return word, target_word_labels, clipped_target_lengths

    def process_dismatched(self, dismatched):
        dismatched_words = tf.string_split(tf.expand_dims(dismatched, -1), delimiter=' ')
        dismatched_words_shape = [1, tf.maximum(tf.to_int64(self.config.MAX_DISMATCHED_PARTS), dismatched_words.dense_shape[1])]
        sparse_dismatched = tf.sparse.SparseTensor(indices=dismatched_words.indices,
                                                      values=dismatched_words.values,
                                                      dense_shape=dismatched_words_shape)
        dense_dismatched = tf.reshape(tf.sparse.to_dense(sp_input=sparse_dismatched,
                                                           default_value=Common.PAD), [-1])
        index_of_blank = tf.where(tf.equal(dense_dismatched, Common.PAD))
        dismatched_length = tf.reduce_min(index_of_blank)
        dense_dismatched = dense_dismatched[:self.config.MAX_DISMATCHED_PARTS]
        dismatched_length = tf.clip_by_value(dismatched_length, clip_value_min=0, clip_value_max=self.config.MAX_DISMATCHED_PARTS)
        dismatched_word_labels = self.subtoken_table.lookup(dense_dismatched)  # (max_time)
        return dismatched, dismatched_word_labels, dismatched_length

    def process_contexts(self, contexts):
        contexts = tf.reshape(contexts, [-1])
        all_contexts = tf.string_split(contexts, delimiter=' ', skip_empty=False).values
        all_contexts_padded = tf.concat([all_contexts, [self.context_pad]], axis=-1)
        dense_split_contexts = self.split_contexts(all_contexts)
        if not self.is_evaluating and self.config.RANDOM_CONTEXTS:                                     # train
            contexts = self.separate_random_sample(dense_split_contexts, all_contexts_padded)
        else:
            contexts = dense_split_contexts

        return contexts

    def separate_random_sample(self, dense_split_contexts, all_contexts_padded):
        index_of_blank_context = tf.where(tf.equal(all_contexts_padded, self.context_pad))
        num_contexts_per_example = tf.reduce_min(index_of_blank_context)
        safe_limit = tf.cast(tf.maximum(num_contexts_per_example, self.config.MAX_CONTEXTS), tf.int32)
        rand_indices = tf.random_shuffle(tf.range(safe_limit))[:self.config.MAX_CONTEXTS]
        contexts = tf.gather(dense_split_contexts, rand_indices)  # (max_contexts,)
        return contexts

    def split_contexts(self, contexts):
        split_contexts = tf.string_split(contexts, delimiter=',', skip_empty=False)

        sparse_split_contexts = tf.sparse.SparseTensor(indices=split_contexts.indices,
                                                       values=split_contexts.values,
                                                       dense_shape=[self.config.MAX_CONTEXTS, 3])
        dense_split_contexts = tf.reshape(
            tf.sparse.to_dense(sp_input=sparse_split_contexts, default_value=Common.PAD),
            shape=[self.config.MAX_CONTEXTS, 3])
        return dense_split_contexts   # (batch, max_contexts, 3)

    def process_context_source_string(self, dense_split_contexts):
        # dense_split_contexts   (max_contexts, 3)
        path_source_strings = tf.slice(dense_split_contexts, [0, 0], [self.config.MAX_CONTEXTS,
                                                                      1])   # (max_contexts, 1) get the first element in the triple
        flat_source_strings = tf.reshape(path_source_strings, [-1])  # (max_contexts)
        split_source = tf.string_split(flat_source_strings, delimiter='|', skip_empty=False)

        sparse_split_source = tf.sparse.SparseTensor(indices=split_source.indices, values=split_source.values,
                                                     dense_shape=[self.config.MAX_CONTEXTS, tf.maximum(tf.to_int64(self.config.MAX_NAME_PARTS),
                                                                                                       split_source.dense_shape[1])])
        dense_split_source = tf.sparse.to_dense(sp_input=sparse_split_source,
                                                default_value=Common.PAD)  # (max_contexts, max_name_parts)

        # tf.Print(tf.shape(dense_split_source),[tf.shape(dense_split_source)], message='dense_split_source')

        dense_split_source = tf.slice(dense_split_source, [0, 0], [-1, self.config.MAX_NAME_PARTS])
        path_source_indices = self.subtoken_table.lookup(dense_split_source)  # (max_contexts, max_name_parts)
        path_source_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_source, Common.PAD), tf.int32),
                                            -1)  # (max_contexts)
        return path_source_strings, path_source_indices, path_source_lengths

    def process_context_path_string(self, dense_split_contexts):
        path_strings = tf.slice(dense_split_contexts, [0, 1], [self.config.MAX_CONTEXTS, 1])
        flat_path_strings = tf.reshape(path_strings, [-1])

        split_path = tf.string_split(flat_path_strings, delimiter='|', skip_empty=False)
        sparse_split_path = tf.sparse.SparseTensor(indices=split_path.indices, values=split_path.values,
                                                   dense_shape=[self.config.MAX_CONTEXTS, self.config.MAX_PATH_LENGTH])   # if path length > max_path_length wrong
        dense_split_path = tf.sparse.to_dense(sp_input=sparse_split_path,
                                              default_value=Common.PAD)  # (batch, max_contexts, max_path_length)

        node_indices = self.node_table.lookup(dense_split_path)  # (max_contexts, max_path_length)
        path_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_path, Common.PAD), tf.int32), -1)  # (max_contexts)

        # tf.Print(tf.shape(path_strings),[tf.shape(path_strings)], message='path_strings')
        # tf.Print(tf.shape(path_lengths),[tf.shape(path_lengths)], message='path_lengths')
        return path_strings, node_indices, path_lengths

    def process_context_target_string(self, dense_split_contexts):
        path_target_strings = tf.slice(dense_split_contexts, [0, 2], [self.config.MAX_CONTEXTS, 1])  # (max_contexts, 1)
        flat_target_strings = tf.reshape(path_target_strings, [-1])  # (max_contexts)
        split_target = tf.string_split(flat_target_strings, delimiter='|',
                                       skip_empty=False)  # (max_contexts, max_name_parts)
        sparse_split_target = tf.sparse.SparseTensor(indices=split_target.indices, values=split_target.values,
                                                     dense_shape=[self.config.MAX_CONTEXTS,
                                                                  tf.maximum(tf.to_int64(self.config.MAX_NAME_PARTS),
                                                                             split_target.dense_shape[1])])
        dense_split_target = tf.sparse.to_dense(sp_input=sparse_split_target,
                                                default_value=Common.PAD)  # (max_contexts, max_name_parts)
        dense_split_target = tf.slice(dense_split_target, [0, 0], [-1, self.config.MAX_NAME_PARTS])
        path_target_indices = self.subtoken_table.lookup(dense_split_target)  # (max_contexts, max_name_parts)
        path_target_lengths = tf.reduce_sum(tf.cast(tf.not_equal(dense_split_target, Common.PAD), tf.int32),
                                            -1)  # (max_contexts)
        return path_target_strings, path_target_indices, path_target_lengths

    def reset(self, sess):
        sess.run(self.reset_op)

    def get_output(self):
        return self.output_tensors

    def compute_output(self):
        dataset = tf.data.experimental.CsvDataset(self.file_path, record_defaults=[tf.string, tf.string, tf.string, tf.string, tf.string, tf.string, tf.string, tf.string], header=True, buffer_size=self.config.CSV_BUFFER_SIZE)   #automaticly fill with records_defaults
        if not self.is_evaluating:
            if self.config.SAVE_EVERY_EPOCHS > 1:
                dataset = dataset.repeat(self.config.SAVE_EVERY_EPOCHS)
            dataset = dataset.shuffle(self.config.SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=self.process_dataset, batch_size=self.batch_size,
            num_parallel_batches=self.config.READER_NUM_PARALLEL_BATCHES))
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        self.iterator = dataset.make_initializable_iterator()
        self.reset_op = self.iterator.initializer
        return self.iterator.get_next()