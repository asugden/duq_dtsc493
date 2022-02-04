import numpy as np
import pandas as pd
import sklearn.model_selection  # pip install scikit-learn

import week4.class_wrapper


if __name__ == '__main__':
    path = 'data/winequality-white.csv'
    df = pd.read_csv(path, delimiter=';')
    df['good'] = df['quality'] > 5

    features = df.drop(columns=['good', 'quality']).values
    labels = df['good'].values

    train_features, test_features, train_labels, test_labels = \
        sklearn.model_selection.train_test_split(features, labels, test_size=0.3)

    cw = week4.class_wrapper.ClassifierWrapper()
    cw.train(train_features, train_labels)
    print(cw.assess(test_features, test_labels, 'percent_correct'))
    cw.save('data/wine_model.xgb')

    cw = week4.class_wrapper.ClassifierWrapper()
    cw.train(train_features, train_labels, decision_tree=True)
    print(cw.assess(test_features, test_labels, 'percent_correct'))
    # test_preds = cw.apply(test_features)
    # test_preds = test_preds[:, 1] >= 0.5

    # correct = 1 - np.abs(test_preds.astype(int) - test_labels.astype(int))
    # print(correct.sum()/len(correct))
