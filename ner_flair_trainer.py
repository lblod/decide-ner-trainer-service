# based on https://flairnlp.github.io/docs/tutorial-training/how-to-train-sequence-tagger

import os
import fire
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from huggingface_hub import create_repo, upload_folder
from dot_env import load_dotenv


load_dotenv()


def train_flair(data_folder: str, hf_model_name: str, transformer_embedding_model: str = "xlm-roberta-large"):
    columns = {0: "text", 3: "ner"}

    corpus = ColumnCorpus(
        data_folder,
        column_format=columns,
        train_file="train.txt",
        dev_file="dev.txt",
        test_file="test.txt",
        encoding="utf-8",
    )

    label_type = "ner"

    label_dict = corpus.make_label_dictionary(
        label_type=label_type, add_unk=False)

    embeddings = TransformerWordEmbeddings(model=transformer_embedding_model,
                                           layers="-1",
                                           subtoken_pooling="first",
                                           fine_tune=True,
                                           use_context=True,
                                           )

    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type="ner",
                            use_crf=False,
                            use_rnn=False,
                            reproject_embeddings=False,
                            )

    trainer = ModelTrainer(tagger, corpus)

    trainer.fine_tune(f"resources/taggers/{hf_model_name}",
                      learning_rate=5.0e-6,
                      mini_batch_size=4,
                      # remove this parameter to speed up computation if you have a big GPU
                      mini_batch_chunk_size=1,
                      )

    repo_id = f"{os.getenv('HUGGINGFACE_HANDLE')}/decide-{hf_model_name}"

    create_repo(repo_id, private=False, exist_ok=True)

    upload_folder(
        repo_id=repo_id,
        folder_path=f"resources/taggers/{hf_model_name}",
        path_in_repo=".",
    )

    print("Pushed model to:", f"https://huggingface.co/{repo_id}")


if __name__ == '__main__':
    fire.Fire(train_flair)
