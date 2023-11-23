import joblib
from utils import preprocess

def load_model():
    vectorizer_filename = './model/count_vectorizer.joblib'
    model_filename = './model/bayes.model'
    vectorizer = joblib.load(vectorizer_filename)
    model = joblib.load(model_filename)
    return vectorizer, model

vectorizer , model = load_model()

id2label = {
    0: "CLEAN",
    1: "OFFENSIVE",
    2: "HATE"
}


def bayes_sentiment_classification(text :str):
    text_preprocess = preprocess(text)
    text_vectorizer = vectorizer.transform([text_preprocess]) 
    label_pred = model.predict_proba(text_vectorizer)[0]
    list_label = ["CLEAN", 'HATE']
    result_dict = dict(zip(list_label, label_pred))
    print(result_dict)
    return result_dict

bayes_sentiment_classification("xin chao")