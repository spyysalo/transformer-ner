import os
import math
import json

from transformers import AutoConfig, AutoTokenizer, TFAutoModel
from transformers.optimization_tf import AdamWeightDecay

from util import logger, timed


@timed
def load_pretrained(model_name, cache_dir=None):
    from transformers import TFBertModel

    # TODO confirm that output_hidden_states=True is supported by all
    # relevant models
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        output_hidden_states=True
    )
    # `use_fast=False` here because the "fast" tokenizer encode()
    # doesn't accept list of strings as an argument. TODO: rework the
    # use of encode() to be compatible with the fast tokenizers.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir,
        use_fast=False
    )
    model = TFAutoModel.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir
    )

    # Transformers doesn't support saving models wrapped in keras
    # (https://github.com/huggingface/transformers/issues/2733) at the
    # time of this writing. As a workaround, use the main layer
    # instead of the model. As the main layer has different names for
    # different models (TFBertModel.bert, TFRobertaModel.roberta,
    # etc.), this has to check which model we're dealing with.
    if isinstance(model, TFBertModel):
        model = model.bert
    else:
        raise NotImplementedError(f'{model.__class__.__name__}')

    return model, tokenizer, config


def get_optimizer(lr, epochs, batch_size, warmup_proportion,
                  num_train_examples):
    from transformers.optimization_tf import create_optimizer

    steps_per_epoch = math.ceil(num_train_examples / batch_size)
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = math.floor(num_train_steps * warmup_proportion)

    # Mostly defaults from transformers.optimization_tf
    optimizer, lr_schedule = create_optimizer(
        lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        min_lr_ratio=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay_rate=0.01,
        power=1.0,
    )
    return optimizer, lr_schedule


def build_ner_model(pretrained_model, num_labels, seq_len):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Dropout, Dense, Average, Concatenate
    
    input_ids = Input(
        shape=(seq_len,), dtype='int32', name='input_ids')
    token_type_ids = Input(
        shape=(seq_len,), dtype='int32', name='token_type_ids')
    attention_mask = Input(
        shape=(seq_len,), dtype='int32', name='attention_mask')
    inputs = [input_ids, attention_mask, token_type_ids]

    pretrained_outputs = pretrained_model(inputs)
    sequence_output = pretrained_outputs[0]
    encoder_outputs = pretrained_outputs[2]

    # concatenate output of the last four layers
    sequence_output = Concatenate()(encoder_outputs[-4:])
    logger.info(f'concatenated output shape: {sequence_output.shape}')

    sequence_output = Dropout(0.1)(sequence_output)

    ner_output = Dense(
        num_labels,
        activation='softmax'
    )(sequence_output)

    ner_model = Model(
        inputs=inputs,
        outputs=[ner_output]
    )
    return ner_model


def _ner_model_path(ner_model_dir):
    return os.path.join(ner_model_dir, 'model.hdf5')


def _ner_labels_path(ner_model_dir):
    return os.path.join(ner_model_dir, 'labels.txt')


#def _ner_model_config_path(ner_model_dir):
#    return os.path.join(ner_model_dir, 'model_config.json')


def _ner_decoder_path(ner_model_dir):
    return os.path.join(ner_model_dir, 'decoder.json')


def save_ner_model(directory, ner_model, decoder, tokenizer, labels, config):
    os.makedirs(directory, exist_ok=True)
    ner_model.save(_ner_model_path(directory))
    decoder.save(_ner_decoder_path(directory))
    config.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    with open(_ner_labels_path(directory), 'w') as out:
        for label in labels:
            print(label, file=out)


def _get_custom_objects():
    """Get custom objects for loading saved model."""
    return  {
        'AdamWeightDecay': AdamWeightDecay,
    }


def load_labels(path):
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.isspace() or not line:
                continue
            if line in labels:
                raise ValueError(f'duplicate value {line} in {path}')
            labels.append(line)
    return labels


def load_ner_model(ner_model_dir):
    from tensorflow.keras.models import load_model
    from viterbi import ViterbiDecoder

    config = AutoConfig.from_pretrained(ner_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        ner_model_dir,
        config=config,
        use_fast=False
    )
    ner_model = load_model(
        _ner_model_path(ner_model_dir),
        custom_objects=_get_custom_objects()
    )
    decoder = ViterbiDecoder.load(_ner_decoder_path(ner_model_dir))
    labels = load_labels(_ner_labels_path(ner_model_dir))
    return ner_model, decoder, tokenizer, labels, config
