# Transformer NER

NER using Transformers, TensorFlow, and Keras

## Quickstart

Install requirements

```
python -m pip install -r requirements.txt
```

Get data

```
./scripts/get-turku-ner.sh
```

Run evaluation

```
python train.py \
       --model_name TurkuNLP/bert-base-finnish-cased-v1 \
       --labels data/turku-ner/labels.txt \
       --train_data data/turku-ner/train.tsv \
       --dev_data data/turku-ner/dev.tsv
```
