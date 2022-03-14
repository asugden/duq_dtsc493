import numpy as np
import pandas as pd
import sklearn.model_selection  # pip install scikit-learn

import week7.parxgb


if __name__ == '__main__':
    path = 'data/winequality-white.csv'
    df = pd.read_csv(path, delimiter=';')
    df['good'] = df['quality'] > 5

    features = df.drop(columns=['good', 'quality'])
    labels = df['quality']

    train_features, test_features, train_labels, test_labels = \
        sklearn.model_selection.train_test_split(
            features, labels, test_size=0.3)

    cw = week7.parxgb.ParameterizedXGBoost()
    cw.train(train_features, train_labels, categorical=True)
    pred_labels = cw.apply(test_features)

    print(np.mean(np.square(pred_labels.astype(float) - test_labels.astype(float))))
