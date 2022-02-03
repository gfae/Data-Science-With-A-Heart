import pandas as pd
from os.path import dirname, join
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
sns.set(style='darkgrid')

def ML():
    currentDir = dirname(__file__)
    filePath = join(currentDir, "./heart.csv")
    df = pd.read_csv(filePath)
    df.columns = ['Age', 'Sex', 'Chest_pain_type', 'Resting_bp',
                  'Cholesterol', 'Fasting_bs', 'Resting_ecg',
                  'Max_heart_rate', 'Exercise_induced_angina',
                  'ST_depression', 'ST_slope', 'Num_major_vessels',
                  'Thallium_test', 'Condition']

    X = df.drop(['Condition'], axis=1)
    y = df.Condition
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    def get_normalization(X):
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        return X_normalized

    def get_model_accuracy(model, X_test, y_test):
        """
        Return the mean accuracy of model on X_test and y_test
        """
        model_acc = model.score(X_test, y_test)
        return model_acc

    X_train = get_normalization(X_train);
    X_test = get_normalization(X_test);
    # X_fit = get_normalization(finaldf)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    df2 = pd.DataFrame([[52,1,2,172,199,1,1,162,0,0.5,2,0,3],[48,0,2,130,275,0,1,139,0,0.2,2,0,2],[57,0,1,130,236,0,0,174,0,0,1,1,2],[68,1,0,144,193,1,1,141,0,3.4,1,2,3],[57,0,0,120,354,0,1,163,1,0.6,2,0,2]], columns=['Age', 'Sex', 'Chest_pain_type', 'Resting_bp',
                                        'Cholesterol', 'Fasting_bs', 'Resting_ecg',
                                        'Max_heart_rate', 'Exercise_induced_angina',
                                        'ST_depression', 'ST_slope', 'Num_major_vessels',
                                        'Thallium_test'])
    X_fit = get_normalization(df2.values)
    prediction = lr.predict(df2.values)
    logreg_acc = get_model_accuracy(lr, X_test, y_test)
    # print(f'Logistic Regression Accuracy: {logreg_acc:.4}')

    return (prediction)

print(ML())