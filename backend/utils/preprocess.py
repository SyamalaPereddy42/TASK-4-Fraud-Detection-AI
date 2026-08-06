import numpy as np

def preprocess_input(data, scaler):
    """
    Preprocess the incoming patient data using the saved scaler.
    """

    values = np.array(data).reshape(1, -1)

    values = scaler.transform(values)

    return values