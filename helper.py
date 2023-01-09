import getopt
import pandas as pd
from sklearn.model_selection import train_test_split

import utils


def handle_arguments(argv):
    is_testing = False
    is_training = False
    is_loading_model = False
    argv = argv[1:]
    long_options = ["train", "test", "load_model"]
    try:
        opts, args = getopt.getopt(argv, "", long_options)
        for opt, arg in opts:
            if opt in ["--train"]:
                is_training = True
            elif opt in ["--test"]:
                is_testing = True
            elif opt in ["--load_model"]:
                is_loading_model = True
    except:
        print("Argument error.")
        print("Use argument \"--train\" for training")
        print("Or argument \"--test\" for testing")
        print("Or argument \"--load_model\" for loading the latest model")
        exit()

    if not is_training and not is_testing:
        print("Must be training, testing or both.")
        print("Use argument \"--train\" for training")
        print("Or argument \"--test\" for testing")
        print("Or argument \"--load_model\" for loading latest model")
        exit()

    return is_testing, is_training, is_loading_model


def read_data(fake_path: str, true_path: str):
    df_fake = pd.read_csv(fake_path, index_col=0)
    df_true = pd.read_csv(true_path, index_col=0)
    df_fake.dropna(inplace=True)
    df_true.dropna(inplace=True)
    return df_fake, df_true


def add_labels(df_fake: pd.DataFrame, df_true: pd.DataFrame, fake_label: int, true_label: int):
    df_fake["label"] = [fake_label] * len(df_fake)
    df_true["label"] = [true_label] * len(df_true)


def concat_data(df_fake: pd.DataFrame, df_true: pd.DataFrame):
    df = pd.concat([df_fake, df_true])
    df.reset_index(inplace=True, drop=True)
    return df


def split_data(df: pd.DataFrame):
    # 60% train, 20% val, 20% test
    train_df, test_df = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=utils.RANDOM_SEED)
    train_df, val_df = train_test_split(train_df, stratify=train_df["label"], test_size=0.25,
                                        random_state=utils.RANDOM_SEED)
    return train_df, val_df, test_df


def add_tokens_and_attention_masks(df: pd.DataFrame, tokenizer, col1="tokens", col2="attention_masks"):
    train_tokens = tokenizer(df["text"].tolist(), padding=utils.PADDING, truncation=utils.TRUNCATION,
                             return_tensors="pt", max_length=512)
    train_tokens = train_tokens.data
    df[col1] = list(train_tokens["input_ids"])
    df[col2] = list(train_tokens["attention_mask"])


def save_file(args: dict, file_name: str):
    results_df = pd.DataFrame(args, columns=list(args.keys()))
    results_df.to_csv(file_name, index=False)



