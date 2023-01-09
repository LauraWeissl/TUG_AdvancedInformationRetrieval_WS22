import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import sklearn
import matplotlib.pyplot as plt
import json

import helper
import utils


def classify_emotions(df: pd.DataFrame):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_dataloader = torch.utils.data.DataLoader(df["text"].tolist(), batch_size=8)
    classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion',
                          return_all_scores=True, device=device)
    predictions = []
    for text in tqdm(test_dataloader):
        prediction = classifier(text, max_length=512, truncation=True)
        predictions.extend(prediction)

    save_args = {"predictions": predictions}
    helper.save_file(save_args, utils.RESULTS_PATH + "emotion_predictions.csv")


def classify_sentiment(df: pd.DataFrame):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_dataloader = torch.utils.data.DataLoader(df["text"].tolist(), batch_size=8)
    classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment",
                          return_all_scores=True, device=device)
    predictions = []
    for text in tqdm(test_dataloader):
        prediction = classifier(text, max_length=512, truncation=True)
        predictions.extend(prediction)

    save_args = {"predictions": predictions}
    helper.save_file(save_args, utils.RESULTS_PATH + "sentiment_predictions.csv")


def plot_and_print_metrics(predictions_df):
    # metrics
    cr = sklearn.metrics.classification_report(predictions_df["label"], predictions_df["predictions"],
                                               target_names=["Fake", "True"])

    # Confusion matrix
    cm = sklearn.metrics.confusion_matrix(predictions_df["label"],
                                          predictions_df["predictions"])
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                                  display_labels=["Fake", "True"])
    disp.plot(cmap="YlGnBu")
    plt.savefig(utils.PLOT_PATH + "confusion_matrix.png")
    plt.show()

    # print results
    print("-" * 20)
    print("Confusion Matrix")
    print(cm)
    print("-" * 20)
    print("Metrics")
    print(cr)


def plot_loss_and_accuracy(df: pd.DataFrame):
    plt.plot(df["train_loss"])
    plt.plot(df["val_loss"])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(utils.PLOT_PATH + "loss.png")
    plt.show()

    plt.plot(df["train_acc"])
    plt.plot(df["val_acc"])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(utils.PLOT_PATH + "accuracy.png")
    plt.show()


def analyze_emotion_classifications(df: pd.DataFrame):
    total_emotions = {"sadness": df["sadness"].mean(), "joy": df["joy"].mean(),
                      "love": df["love"].mean(), "anger": df["anger"].mean(),
                      "fear": df["fear"].mean(), "surprise": df["surprise"].mean()}
    print(json.dumps(total_emotions, indent=4))


def analyze_sentiment_classifications(df: pd.DataFrame):
    total_sentiments = {"negative": df["negative"].mean(), "neutral": df["neutral"].mean(),
                        "positive": df["positive"].mean()}
    print(json.dumps(total_sentiments, indent=4))


def add_separate_sentiment_cols(df: pd.DataFrame):
    negative = []
    neutral = []
    positive = []
    for index, row in df.iterrows():
        json_str = row["sentiment"].replace("\'", "\"")
        sentiments = json.loads(json_str)
        for s in sentiments:
            if s["label"] == "LABEL_0":
                negative.append(s["score"])
            elif s["label"] == "LABEL_1":
                neutral.append(s["score"])
            elif s["label"] == "LABEL_2":
                positive.append(s["score"])

    df["negative"] = negative
    df["neutral"] = neutral
    df["positive"] = positive


def create_df_with_separate_emotion_cols(df: pd.DataFrame):
    total_emotions = {"sadness": [], "joy": [], "love": [], "anger": [],
                      "fear": [], "surprise": []}
    for index, row in df.iterrows():
        json_str = row["emotion"].replace("\'", "\"")
        emotions = json.loads(json_str)
        for e in emotions:
            total_emotions[e["label"]].append(e["score"])

    emotion_df = pd.DataFrame(total_emotions)
    new_df = pd.concat([df, emotion_df], axis=1)
    return new_df


def get_filtered_df(df: pd.DataFrame, col_filter: str, cols: list):
    filtered_df = df
    for col in cols:
        filtered_df = filtered_df[filtered_df[col_filter] > filtered_df[col]]
    return filtered_df



if __name__ == '__main__':
    # read data
    df_fake, df_true = helper.read_data(utils.FAKE_PATH, utils.TRUE_PATH)

    # add labels
    helper.add_labels(df_fake, df_true, 0, 1)

    # concat data
    df = helper.concat_data(df_fake, df_true)

    # data splitting
    # 60% train, 20% val, 20% test
    train_df, val_df, test_df = helper.split_data(df)

    # classify emotions and sentiments
    classify_emotions(test_df)
    classify_sentiment(test_df)

    predictions_df = pd.read_csv(utils.RESULTS_PATH + "test_predictions_final.csv")
    predictions_df["label"] = test_df["label"].tolist()

    sentiment_df = pd.read_csv(utils.RESULTS_PATH + "sentiment_predictions_final.csv")
    predictions_df["sentiment"] = sentiment_df["predictions"]

    emotion_df = pd.read_csv(utils.RESULTS_PATH + "emotion_predictions_final.csv")
    predictions_df["emotion"] = emotion_df["predictions"]

    add_separate_sentiment_cols(predictions_df)
    predictions_df = create_df_with_separate_emotion_cols(predictions_df)

    # true positives
    print("True positive")
    preds = predictions_df[(predictions_df["label"] == predictions_df["predictions"])]
    preds = preds[preds["label"] == 1]
    analyze_sentiment_classifications(preds)
    analyze_emotion_classifications(preds)

    # true negatives
    print("True negative")
    preds = predictions_df[(predictions_df["label"] == predictions_df["predictions"])]
    preds = preds[preds["label"] == 0]
    analyze_sentiment_classifications(preds)
    analyze_emotion_classifications(preds)

    # false positives
    print("False positive")
    preds = predictions_df[(predictions_df["label"] != predictions_df["predictions"])]
    preds = preds[preds["label"] == 0]
    analyze_sentiment_classifications(preds)
    analyze_emotion_classifications(preds)

    # false negatives
    print("False negative")
    preds = predictions_df[(predictions_df["label"] != predictions_df["predictions"])]
    preds = preds[preds["label"] == 1]
    analyze_sentiment_classifications(preds)
    analyze_emotion_classifications(preds)

    print("Negative accuracy")
    filtered_df = get_filtered_df(predictions_df, "negative", ["positive", "neutral"])
    cr = sklearn.metrics.classification_report(filtered_df["label"], filtered_df["predictions"],
                                               target_names=["Fake", "True"])
    print(cr)

    print("Neutral accuracy")
    filtered_df = get_filtered_df(predictions_df, "neutral", ["positive", "negative"])
    cr = sklearn.metrics.classification_report(filtered_df["label"], filtered_df["predictions"],
                                               target_names=["Fake", "True"])
    print(cr)

    print("Positive accuracy")
    filtered_df = get_filtered_df(predictions_df, "positive", ["negative", "neutral"])
    cr = sklearn.metrics.classification_report(filtered_df["label"], filtered_df["predictions"],
                                               target_names=["Fake", "True"])
    print(cr)

    print("Anger accuracy")
    filtered_df = get_filtered_df(predictions_df, "anger", ["sadness", "joy", "love", "fear", "surprise"])
    cr = sklearn.metrics.classification_report(filtered_df["label"], filtered_df["predictions"],
                                               target_names=["Fake", "True"])
    print(cr)

    print("Sadness accuracy")
    filtered_df = get_filtered_df(predictions_df, "sadness", ["anger", "joy", "love", "fear", "surprise"])
    cr = sklearn.metrics.classification_report(filtered_df["label"], filtered_df["predictions"],
                                               target_names=["Fake", "True"])
    print(cr)

    print("Sadness accuracy")
    filtered_df = get_filtered_df(predictions_df, "joy", ["anger", "sadness", "love", "fear", "surprise"])
    cr = sklearn.metrics.classification_report(filtered_df["label"], filtered_df["predictions"],
                                               target_names=["Fake", "True"])
    print(cr)

    print("Love accuracy")
    filtered_df = get_filtered_df(predictions_df, "love", ["anger", "sadness", "joy", "fear", "surprise"])
    cr = sklearn.metrics.classification_report(filtered_df["label"], filtered_df["predictions"],
                                               target_names=["Fake", "True"])
    print(cr)

    print("Fear accuracy")
    filtered_df = get_filtered_df(predictions_df, "fear", ["anger", "sadness", "joy", "love", "surprise"])
    cr = sklearn.metrics.classification_report(filtered_df["label"], filtered_df["predictions"],
                                               target_names=["Fake", "True"])
    print(cr)

    print("Surprise accuracy")
    filtered_df = get_filtered_df(predictions_df, "surprise", ["anger", "sadness", "joy", "love", "fear"])
    cr = sklearn.metrics.classification_report(filtered_df["label"], filtered_df["predictions"],
                                               target_names=["Fake", "True"])
    print(cr)





