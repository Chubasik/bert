import tokenization
import modeling
import tensorflow as tf
from bert_classification_processor import ClassificationProcessor
import os
from run_classifier import model_fn_builder
from run_classifier import file_based_convert_examples_to_features
from run_classifier import file_based_input_fn_builder

import pandas as pd

class BertClassifier:
    def __init__(self,
                 bert_config_file=None,
                 vocab_file=None,
                 output_dir=None,
                 init_checkpoint=None,
                 do_lower_case=True,
                 max_seq_length=128,
                 train_batch_size=32,
                 learning_rate=5e-5,
                 num_train_epochs=3.0,
                 warmup_proportion=0.1,
                 save_checkpoints_steps=1000,
                 iterations_per_loop=1000,
                 use_tpu=False,
                 tpu_name=None,
                 tpu_zone=None,
                 gcp_project=None,
                 master=None,
                 num_tpu_cores=8):
        self._bert_config_file = bert_config_file
        self._vocab_file = vocab_file
        self._output_dir = output_dir
        self._init_checkpoint = init_checkpoint
        self._do_lower_case = do_lower_case
        self._max_seq_length = max_seq_length
        self._train_batch_size = train_batch_size
        self._learning_rate = learning_rate
        self._num_train_epochs = num_train_epochs
        self._warmup_proportion = warmup_proportion
        self._save_checkpoints_steps = save_checkpoints_steps
        self._iterations_per_loop = iterations_per_loop
        self._use_tpu = use_tpu
        self._tpu_name = tpu_name
        self._tpu_zone = tpu_zone
        self._gcp_project = gcp_project
        self._master = master
        self._num_tpu_cores = num_tpu_cores

        tokenization.validate_case_matches_checkpoint(self._do_lower_case, self._init_checkpoint)

        self._bert_config = modeling.BertConfig.from_json_file(self._bert_config_file)

        if self._max_seq_length > self._bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self._max_seq_length, self._bert_config.max_position_embeddings)
            )

        tf.gfile.MakeDirs(self._output_dir)

        self._processor = ClassificationProcessor()

        self._tokenizer = tokenization.FullTokenizer(
            vocab_file=self._vocab_file, do_lower_case=self._do_lower_case)

        tpu_cluster_resolver = None
        if self._use_tpu and self._tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                self._tpu_name, zone=self._tpu_zone, project=self._gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        self._run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=self._master,
            model_dir=self._output_dir,
            save_checkpoints_steps=self._save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self._iterations_per_loop,
                num_shards=self._num_tpu_cores,
                per_host_input_for_training=is_per_host))

        train_examples = None
        num_train_steps = None
        num_warmup_steps = None


    def fit(self, X, y, batch_size):
        df_train_bert = pd.DataFrame(
            {'id': [i for i in range(len(X))],
             'label': y,
             'a': ['a'] * len(X),
             'text': X.replace(r'\n', ' ',regex=True)}
        )
        df_train_bert.to_csv(os.path.join(self._data_dir, 'train.tsv'), sep='\t', index=False, header=False)
        label_list = df_train_bert.label.unique().tolist()
        label_decoder = {idx: label for idx, label in enumerate(label_list)}

        train_examples = self._processor.get_train_examples(self._data_dir)
        num_train_steps = int(len(train_examples) / self._train_batch_size * self._num_train_epochs)
        num_warmup_steps = int(num_train_steps * self._warmup_proportion)

        model_fn = model_fn_builder(
            bert_config=self._bert_config,
            num_labels=len(label_list),
            init_checkpoint=self._init_checkpoint,
            learning_rate=self._learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=self._use_tpu,
            use_one_hot_embeddings=self._use_tpu
        )

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=self._use_tpu,
            model_fn=model_fn,
            config=self._run_config,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            predict_batch_size=batch_size
        )

        train_file = os.path.join(self._output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, self._max_seq_length, self._tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=self._max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)










