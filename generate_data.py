# ## Load Libraries

import numpy as np


def gen_data(classes=2, imbalance=0.5, n=10):
    raw_data = None
    num_columns = 10
    
    for i in range(0, classes):
        # Calculate how many rows to create
        class_n = round(n*(1-imbalance))
        if i > 0:
            class_n = max(round(n*(imbalance/(classes-1))),3)
        
        # Create a signature for this class
        class_signature = np.random.rand(1, num_columns) * 0.5
        
        # Create data for this class
        # Make some noise
        class_data = np.random.rand(class_n, num_columns) * float(classes)
        
        # Add the class signal
        class_data = class_data + class_signature
        
        # Add the class column
        class_col = np.vstack([i]*class_n)
        class_data = np.hstack((class_col, class_data))
        
        # Add it to the data set
        if raw_data is None:
            raw_data = class_data
        else:
            raw_data = np.vstack((raw_data, class_data))
    
    return raw_data
