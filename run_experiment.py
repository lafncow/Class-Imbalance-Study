# ## Load Libraries

from generate_data import gen_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def run(cfg):
    scores = []
    
    # Generate Data
    data_raw = gen_data(cfg["classes"], cfg["imbalance"], cfg["n"])
    
    # Split Data
    data_train, data_test = train_test_split(data_raw)
    
    # Do Treatment
    X_train, y_train = cfg["treatment"].fit_resample(data_train[:, 1:], data_train[:, 0])
    
    # Fit Model
    cfg["model"].fit(X_train, y_train)
    
    # Get Predictions
    y_pred = cfg["model"].predict(data_test[:, 1:])
    
    # Score it
    for m in cfg["metrics"]:
        score = m(data_test[:, 0], y_pred)
        scores.append(score)
    
    
    # Get F1 score
    f1 = f1_score(data_test[:, 0], y_pred)
    # Find F1-optimal threshold
    threshold = f1/2
    # Apply threshold
    y_pred[y_pred < threshold] = 0
    y_pred[y_pred >= threshold] = 1
    # Create Confusion Matrix
    cf_matrix = confusion_matrix(data_test[:, 0], y_pred)
    
    return scores, cf_matrix
