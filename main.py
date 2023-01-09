import utils
import torch
import sys
from os import path
from transformers import DistilBertModel, DistilBertTokenizerFast
import pandas as pd
import torchinfo

from dataset import Dataset
from training import train
from evaluation import evaluate
from config import Config
import helper
import analyzer


from model import FakeDetectorModel

if __name__ == '__main__':

    # argument handling
    # exists on error
    is_testing, is_training, is_loading_model = helper.handle_arguments(sys.argv)

    # read data
    df_fake, df_true = helper.read_data(utils.FAKE_PATH, utils.TRUE_PATH)

    # add labels
    helper.add_labels(df_fake, df_true, 0, 1)

    # concat data
    df = helper.concat_data(df_fake, df_true)

    # data splitting
    # 60% train, 20% val, 20% test
    train_df, val_df, test_df = helper.split_data(df)

    # tokenization
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    print("Tokenize train")
    helper.add_tokens_and_attention_masks(train_df, tokenizer)

    print("Tokenize val")
    helper.add_tokens_and_attention_masks(val_df, tokenizer)

    print("Tokenize test")
    helper.add_tokens_and_attention_masks(test_df, tokenizer)

    train_dataset = Dataset(train_df)
    val_dataset = Dataset(val_df)
    test_dataset = Dataset(test_df)

    # load model
    bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model = FakeDetectorModel(bert)
    config = Config()
    if is_loading_model:
        if not path.exists(utils.MODEL_PATH):
            print("No final model exists.")
            print("You cannot use \"--load_model\"")
            print("Check " + utils.MODEL_PATH)
            exit()
        checkpoint = torch.load(utils.MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])

    # freeze BERT layers
    for param in model.bert.parameters():
        param.requires_grad = False

    # print model summary
    torchinfo.summary(model, [(512,), (512,)],
                      batch_dim=0,
                      dtypes=[torch.IntTensor, torch.IntTensor],
                      device="cpu")

    # training
    if is_training:
        train_loss, train_acc, val_loss, val_acc = train(model, train_dataset, val_dataset, config)
        save_args = [train_loss, train_acc, val_loss, val_acc]
        save_names = ["train_loss", "train_acc", "val_loss", "val_acc"]
        save_args = dict(zip(save_names, save_args))
        helper.save_file(save_args, utils.RESULTS_PATH + "train_results.csv")
        results_df = pd.DataFrame(save_args, columns=list(save_args.keys()))
        analyzer.plot_loss_and_accuracy(results_df)

    # evaluation
    if is_testing:
        test_loss, test_acc, predictions = evaluate(model, test_dataset, config)
        save_args = [[test_loss], [test_acc]]
        save_names = ["test_loss", "test_acc"]
        save_args = dict(zip(save_names, save_args))
        helper.save_file(save_args, utils.RESULTS_PATH + "test_results.csv")
        save_args = {"predictions": predictions}
        helper.save_file(save_args, utils.RESULTS_PATH + "test_predictions.csv")
        predictions_df = pd.DataFrame(save_args, columns=list(save_args.keys()))
        predictions_df["label"] = test_df["label"].tolist()
        analyzer.plot_and_print_metrics(predictions_df)
