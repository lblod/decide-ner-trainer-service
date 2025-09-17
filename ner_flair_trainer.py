# based on https://flairnlp.github.io/docs/tutorial-training/how-to-train-sequence-tagger

from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


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
