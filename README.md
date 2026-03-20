# ClauseBERT

## Description
ClauseBERT is a BERT-based sequence classifier designed to categorize paragraphs from commercial contracts into 10 target classes (9 specific risk clauses + 1 "Other" class). It is part of the [ClauseLens](https://github.com/weiss25r/ClauseLens) project and it is trained using the [CUAD](https://huggingface.co/datasets/theatticusproject/cuad/tree/main/CUAD_v1$0) dataset. 

The trained model is available in ONNX and safetensor formats on [HuggingFace](https://huggingface.co/raffaele-terracino/ClauseBERT).

## Tech stack
- Pytorch
- Transformers
- ONNX
- Weights & Biases
- Pandas

## Project Pipeline
The project consists of interactive Jupyter Notebooks and source code written using Pytorch. **Weights & Biases** is used to log experiments losses, metrics and confusion matrices. The full pipeline is as follows:
- **Data Preparation**: ```notebooks/data_collection.ipynb``` processes the raw CUAD data into a clean dataset ready for classification. 
- **Training**: ```notebooks/bert_training.ipynb``` handles data splitting, training, model selection, and evaluation. A total of 3 experiments is performed, each corresponding to a different configuration of hyperparameter and specified via JSON config files.
- **Export**: ```notebooks/export_to_onnx.ipynb``` converts the best model in ONNX format, ready for integration into [ClauseLens](https://github.com/weiss25r/ClauseLens).

## Project Structure
```text
├── configs/                # JSON hyperparameter configs for experiments 1, 2, and 3
├── notebooks/              # Jupyter Notebooks
│   ├── data_collection.ipynb   # Data pre-processing
│   ├── bert_training.ipynb     # Model training and evaluation
│   └── export_to_onnx.ipynb    # ONNX conversion
├── src/                    # Core source code
│   ├── dataset.py          # Pytorch Dataset and DataLoaders
│   ├── model.py            # Model architecture
│   └── trainer.py          # Training logic
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.md
```
## Architecture and training details
The ```ClauseDataset``` class in ```src/dataset.py``` tokenizes the data using a specified tokenizer, preparing it for BERT. The backbone model and tokenizer are imported from the Transformers library.

The ```BertClassifier``` class in ```src/model.py ``` adds a dropout layer and a linear classification head on top of BERT.

Training is handled by the ```BertClassifierTrainer class```, which uses:
- a linear learning rate scheduler with warmup
- cross-entropy loss
- the AdamW optimizer


Two checkpoints are saved:
"last": the final trained model
"best": the model with the lowest validation loss


Validation loss is monitored during training to perform checkpointing. At the end of training, evaluation is performed on the validation set. Metrics, losses, and confusion matrices are logged to **Weights & Biases**.


Model selection is based on the highest F1-score. The selected model is then evaluated on the test set.

## Results
The table below show metrics computed on the test set using the chosen classifier, corresponding to the second configuration of hyperparameters.
| Metric   | Validation Set | Test Set |
|-----------|----------------|----------|
| Precision | 0.9109         | 0.8693   |
| Recall    | 0.8938         | 0.8632   |
| F1-score  | 0.9002         | **0.8647**   |


## Acknowledgments
All rights belong to the authors of the [Contract Understanding Atticus Dataset](https://huggingface.co/datasets/theatticusproject/cuad/tree/main/CUAD_v1$0). The dataset is licensed by a CC-BY 4.0 license.

## References
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)


Base model: [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
