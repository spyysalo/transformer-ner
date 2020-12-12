import math

from util import timed


@timed
def load_pretrained(model_name, cache_dir=None):
    from transformers import AutoConfig, AutoTokenizer, TFAutoModel

    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config,
                                              cache_dir=cache_dir)
    model = TFAutoModel.from_pretrained(model_name, config=config,
                                        cache_dir=cache_dir)

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
    from tensorflow.keras.layers import Input, Dropout, Dense
    
    input_ids = Input(
        shape=(seq_len,), dtype='int32', name='input_ids')
    token_type_ids = Input(
        shape=(seq_len,), dtype='int32', name='token_type_ids')
    attention_mask = Input(
        shape=(seq_len,), dtype='int32', name='attention_mask')
    inputs = [input_ids, attention_mask, token_type_ids]

    pretrained_outputs = pretrained_model(inputs)
    sequence_output = pretrained_outputs[0]

    # TODO consider Dropout here
    ner_output = Dense(
        num_labels,
        activation='softmax'
    )(sequence_output)

    ner_model = Model(
        inputs=inputs,
        outputs=[ner_output]
    )
    return ner_model
