import os
import numpy as np
from joblib import dump, load
from model import generate_model

def get_prediced_class(X):
    X = np.array(X).reshape(1, -1)
    if not os.path.exists('model.joblib'):
        # create model if it doesn't exist
        rf = generate_model()
    else:
        # else load model
        rf = load('model.joblib')

    # predict class
    predicted_class = rf.predict(X)
    
    return predicted_class[0]
