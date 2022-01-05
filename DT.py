import json
import joblib
import numpy as np

def predict(text_df):
    text = text_df.text
    f = open("dictionary.json", "r")
    dict_feature = json.loads(f.read())
    
    feature = extract_features(text, dict_feature)
    model = joblib.load("DecisionTree_classifier.pkl")
    
    return model.predict(feature)

    
def extract_features(feature, dict_feature):
  features_matrix = np.zeros((len(feature), len(dict_feature)))
  docID = 0
  for words in feature:
    for word in words:
      wordID = 0
      for i,d in enumerate(dict_feature):
        if d[0] == word:
          wordID = i
          features_matrix[docID,wordID] = words.count(word)
    docID = docID + 1
  return features_matrix

