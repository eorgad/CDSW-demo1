# Read the fitted model from the file model.pkl
# and define a function that uses the model to
# predict petal width from petal length

import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

def predict(args):
  iris_x = np.reshape(float(args.get('petal_length')), (-1,1))
  result = model.predict(iris_x)
  return result[0][0]