# decide-ner-trainer-service
Service that can be used to download data from labelstudio and to train an NER model with it (currently only flair NER models).

## Dependency sync
Syncing the environment to ensure having the correct (versions of) dependencies.
```
uv sync
```

## Environment variables
Create a .env file with the following variables:
```
LABEL_STUDIO_URL="LABEL_STUDIO_INSTANCE_URL"
LABEL_STUDIO_API_KEY="YOUR_LABEL_STUDIO_API_KEY"
LABEL_STUDIO_PROJECT_ID="ID_OF_LABEL_STUDIO_PROJECT_TO_BE_DOWNLOADED"
LABEL_STUDIO_EXPORT_TYPE="CONLL2003"
HUGGINGFACE_HANDLE="YOUR_HUGGINGFACE_HANDLE"
```

## Downloading data
Download the data using the following command:
```
uv run label_studio_data_downloader.py
```
Options:
- `output_path` *(optional, default: "./data/")*: folder to where downloaded data will be saved.

## Splitting the data
Split the downloaded label studio data in train.txt, dev.txt and test.txt file using the following command:
```
uv run data_splitter.py
```
Options:
- `file_path` **(required)**: path to downloaded .conl2003 file.
- `output_folder_path` *(optional, default: "./data/")*: folder to where resulting train.txt, dev.txt and test.txt files must be saved.
- `train_factor` *(optional, default: 0.6)*: factor of data used for training.
- `dev_factor` *(optional, default: 0.2)*: factor of data used for validation.
- `test_factor` *(optional, default: 0.2)*: factor of data used for testing.
  
The first (train_factor x 100)% will be used for training, the last (test_factor x 100)% will be used for testing, and the middle (dev_factor x 100)% of data for validation. Note that the sum of the train_factor, dev_factor and test_factor must be equal to 1.

## Training NER model
Make sure you are authenticated with Hugging Face, then train a flair NER model using the following command:
```
uv run ner_flair_trainer.py
```
Options:
- `data_folder` **(required)**: path to folder containing train.txt, dev.txt and test.txt files.
- `hf_model_name` **(required)**: name of flair Hugging Face model to be fine-tuned. Suggested models: ner-dutch-large & ner-german-large,
- `transformer_embedding_model` *(transformer_embedding_model, "xlm-roberta-large")*: name of transformer embedding to be used.
- `max_epochs` *(optional, default: 20)*: max epochs of training.
- `learning_rate` *(optional, default: 5.0e-6)*: learning rate.
