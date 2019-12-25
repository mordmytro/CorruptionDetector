import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from classification_model import build_classification_model
from value_model import build_value_model

classification_model = build_classification_model()
value_model = build_value_model()

#test_data = train_data.iloc[int(len(train_data)*0.7):]
#test_labels = targets[int(len(train_data)*0.7):]

test_data = pd.read_csv('data/test_converted.csv').drop('Unnamed: 0', axis=1)

class_predictions = [(0 if prediction[0] > prediction[1] else 1) for prediction in classification_model.predict(test_data)]
predictions = []
for i in range(len(class_predictions)):
    if class_predictions[i] == 0:
        predictions.append(0)
    else:
        predictions.append(value_model.predict(np.array([test_data.values[i]]))[0, 0])

df = pd.DataFrame({'id': [f'test_id{i}' for i in range(len(predictions))], 'target': predictions})
df.to_csv('data/submition.csv', index_label=False)