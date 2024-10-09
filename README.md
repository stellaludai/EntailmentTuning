# [EMNLP 2024] Improve Dense Passage Retrieval with Entailment Tuning

## Repo Structure
This repo follow the structure of [DPR](https://github.com/facebookresearch/DPR), and the retriever training / indexing / testing all follows the original repo. There are three main differences in our repo:
1. We reimplement the faiss with faiss-gpu and enable distributed indexing and retriving in multi-gpu setting. The new implementation significantly accelerates the retrieving process, with each query cost less than 1ms.
2. We exclude the reader training codes in DPR, since it's irrelevant with our work.
3. We added the data-downloading / fine-tuning code for our entailment tuning method.

## QuickStart

### 1. Clone Repo
```bash
git clone https://github.com/stellaludai/EntailmentTuning.git
```

### 2. Install Dependencies
```bash
cd EntailmentTuning
pip install .
```

### 3. Download Data
For testing, at least the wikipedia corpus and qa dataset is required. An example of tesing on nq dataset:
```bash
cd dpr
python data/download_data.py --resource data.wikipedia_split.psgs_w100 [optional --output_dir {your location}]
python data/download_data.py --resource data.retriever.qas.nq [optional --output_dir {your location}]
```
For training, you also need to download the retriever training data, as well as the NLI data for entailment tuning:
```bash
# nq training data
python data/download_data.py --resource data.retriever.nq
# NLI dataset used for entailment tuning
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
wget https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
unzip multinli_1.0.zip
```

### 4. Download Model

For testing, we release our [entailment-tuned retriever](https://huggingface.co/stellaludai/dpr_single_nq_base_entailtuned) for nq on huggingface, which can be directly used for retrieving/testing. This model was trained with an extra entailment tuning stage before the DPR supervised training stage.


## Inference

### 1. Update paths of your downloaded datasets
For running in your own environment, you need to replace the placeholder path with your downloaded data path:
```
In conf/ctx_sources/default_sources.yaml:
/paths/to/your/psgs_w100.tsv

In conf/datasets/retriever_default.yaml:
/path/to/your/nq-test.csv

In retriever_generate_embedding.sh:
/path/to/your/biencoder_model.pt(downloaded from our huggingface repo)

In retriever_validate_full.sh:
/path/to/your/biencoder_model.pt(downloaded from our huggingface repo)

```


### 2. Indexing
You may adjust the gpu settings in the bash file to fit your own needs.

```bash
./retriever_generate_embedding.sh
```

### 3. Testing
```bash
./retriever_validate_full.sh
```

## Training

To be added