from transformers import pipeline
from utils import preprocess

classifier = pipeline("sentiment-analysis", model="model/bert_best_model")

def bert_sentiment_classification(text=""):
    result = {'CLEAN':0, 'HATE': 0}
    text = preprocess(text)
    output_sentiment = classifier(text)[0]
    label_predict = output_sentiment['label']
    score_precit = output_sentiment['score']
    if label_predict == "CLEAN":
        result["CLEAN"] = score_precit
        result['HATE'] = 1-score_precit
    else:
        result["HATE"] = score_precit
        result['CLEAN'] = 1-score_precit
    return result

# print(bert_sentiment_classification("thằng chó này"))
