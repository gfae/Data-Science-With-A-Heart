import pandas as pd
from os.path import dirname, join
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
sns.set(style='darkgrid')
def ML(finaldf):
    currentDir = dirname(__file__)
    filePath = join(currentDir, "./heart.csv")
    # with open(filePath, newline='') as csvfile:
    #     testreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #     for row in testreader:
    df = pd.read_csv(filePath)
    df.columns = ['Age', 'Sex', 'Chest_pain_type', 'Resting_bp',
                  'Cholesterol', 'Fasting_bs', 'Resting_ecg',
                  'Max_heart_rate', 'Exercise_induced_angina',
                  'ST_depression', 'ST_slope', 'Num_major_vessels',
                  'Thallium_test', 'Condition']

    print(df)

    X = df.drop(['Condition'], axis=1)
    y = df.Condition
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)


    print("\nX_train:\n")
    print(X_train.head()) #prints the first 5 elements
    print(X_train.shape) #prints number of rows & columns

    print("\nX_test:\n")
    print(X_test.head())
    print(X_test.shape)

    print("\ny_train:\n")
    print(y_train.head())
    print(y_train.shape)

    print("\ny_test:\n")
    print(y_test.head())
    print(y_test.shape)

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

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    f_pred = lr.predict(finaldf)
    logreg_acc = get_model_accuracy(lr, X_test, y_test)
    print(f'Logistic Regression Accuracy: {logreg_acc:.4}')

    return (f_pred)

# drop row if all data is missing
# can use 'any' to drop if any values are null but not needed as other data can still be used
# will not corrupt the dataset
# data = df.dropna(how='all').shape
#
# # print(data.info())
# # print(data.isnull().sum())
# def condition_ratio(data):
#     """
#     Makes 3 pie charts of 'Condition' values
#     Condition: 0 = Benign, 1 = Malignant
#     First is all data, second is female, third is male
#     - results stores the total count of each condition variable
#     - values stores a list of those values
#     plt.pie creates the pie chart
#         - values are self explanatory (autopct will show the percentage)
#     """
#     fig = plt.figure(figsize=(10, 8))
#     plt.subplot(2,3,2)
#     results = data['Condition'].value_counts()
#     values = [results[0], results[1]]
#     labels = ['Benign', 'Malignant']
#     colours = ('DarkTurquoise', 'Salmon')
#     plt.pie(values, labels = labels, colors=colours, autopct='%1.0f%%')
#     plt.title('Benign vs. Malignant', fontsize=15)
#     """
#     Female condtion ratio pie chart
#     """
#     resultsBySex = data['Condition'].groupby(data['Sex']).value_counts()
#     valuesBySex = [resultsBySex[0], resultsBySex[1]]
#     plt.subplot(2,3,4)
#     plt.pie(valuesBySex[0], labels = labels, colors=colours, autopct='%1.0f%%')
#     plt.title('Female Condtion Ratio', fontsize=15)
#     """"
#     Male condtion ratio pie chart
#     """
#     plt.subplot(2,3,6)
#     plt.pie(valuesBySex[1], labels = labels, colors=colours, autopct='%1.0f%%')
#     plt.title('Male Condition Ratio', fontsize=15)
#
#     plt.show()
#
# def isAgeAFactor(data):
#     """
#     Plotting age and condtion against various factors.
#     Benign, malignant and age variables are used in all the following scatter graphs so variables have been set for them.
#     """
#     benign = data.Condition == 0
#     malignant = data.Condition == 1
#     age_benign = data.Age[benign]
#     age_malignant = data.Age[malignant]
#     b_color = 'DarkTurquoise'
#     m_color = 'Salmon'
#     fig = plt.figure(figsize=(16, 10))
#
#     """
#     Comparing age and resting blood pressure
#
#     subplot(x,y,z) sets the layout of the different graphs within the figure
#         - x is the number of rows
#         - y is the number of columns
#         - z is the position of the graph currently being created/adjusted
#     plt.scatter plots points on the graph using the arguments provided
#         - x axis is designated the age
#         - y axis is designated the factor being plotted
#         - s is the size of the scatter points, a smaller number will allow you to see all the points more clearly but a larger number will highlight crowding better
#         - c is the colour, different colours for malignant and benign
#     """
#     plt.subplot(2,2,1)
#     plt.scatter(x=age_benign, y=data.Resting_bp[benign], s=40, c=b_color)
#     plt.scatter(x=age_malignant, y=data.Resting_bp[malignant], s=40, c=m_color)
#     plt.title('Resting BP vs. age', fontsize=15)
#     plt.legend(['Benign', 'Malignant'])
#     plt.xlabel('age', fontsize=10)
#     plt.ylabel('Resting blood pressure (mmHg)', fontsize=10)
#
#
#     """"
#     Comparing age and cholesterol
#     """
#     plt.subplot(2,2,2)
#     plt.scatter(x=age_benign, y=data.Cholesterol[benign], s=40, c=b_color)
#     plt.scatter(x=age_malignant, y=data.Cholesterol[malignant], s=40, c=m_color)
#     plt.title('Cholesterol vs. age', fontsize=15)
#     plt.legend(['Benign', 'Malignant'])
#     plt.xlabel('age', fontsize=10)
#     plt.ylabel('Serum cholestoral (mg/dL)', fontsize=10)
#
#     """
#     Comparing maximum heart rate and age
#     """
#     plt.subplot(2,2,3)
#     plt.scatter(x=age_benign, y=data.Max_heart_rate[benign], s=40, c=b_color)
#     plt.scatter(x=age_malignant, y=data.Max_heart_rate[malignant], s=40, c=m_color)
#     plt.title('Maximum heart rate vs. age', fontsize=15)
#     plt.legend(['Benign', 'Malignant'])
#     plt.xlabel('age', fontsize=10)
#     plt.ylabel('Max heart rate (bpm)', fontsize=10)
#
#
#     """
#     Comparing ST Depression and age
#
#     When performing an ekg, the flat part between the peaks is called the ST segment,
#     The heart isn't doing anything between pumps so this part should be the same as the baseline
#     When the reading is lower than expected, this is called ST depression and can be a sympton of heart disease.
#     """
#     plt.subplot(2,2,4)
#     plt.scatter(x=age_benign, y=data.ST_depression[benign], s=40, c=b_color)
#     plt.scatter(x=age_malignant, y=data.ST_depression[malignant], s=40, c=m_color)
#     plt.title('ST Depression vs. age', fontsize=15)
#     plt.legend(['Benign', 'Malignant'])
#     plt.xlabel('age', fontsize=10)
#     plt.ylabel('ST Depression', fontsize=10)
#
#     plt.tight_layout()
#     plt.show()
#
#
# condition_ratio(df)
# isAgeAFactor(df)
#
#
