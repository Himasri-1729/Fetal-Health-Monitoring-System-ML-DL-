import pandas as pd
import pickle

# Load ML components
s = pickle.load(open(__file__.replace('predict_ml.py', 'scalar.pkl'), 'rb'))
m = pickle.load(open(__file__.replace('predict_ml.py', 'rf_model.pkl'), 'rb'))
c = pickle.load(open(__file__.replace('predict_ml.py', 'columns.pkl'), 'rb'))

def predict_ml(data_dict):
    df = pd.DataFrame([data_dict])[c]
    transformed = s.transform(df)
    pred = m.predict(transformed)
    return int(pred[0])
