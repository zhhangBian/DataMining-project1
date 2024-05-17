import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./dataset/train.csv')
df.head()

data_num_one = []
data_num_not = []
data_classify = []

features = df.columns
for f in features:
    if df[f].unique().size > 5:
        if not isinstance(df[f].max(), str):
            if df[f].max() <= 1.1:
                data_num_one.append(f)
            else:
                print(f + ": " + str(df[f].max()))
                data_num_not.append(f)
        else:
            data_num_not.append(f)
    else:
        data_classify.append(f)

    # print(f + ": " + str(df[f].unique().size) + " ", sorted(df[f].unique()))
    # print(f + ": " + str(df[f].unique().size) )
print()
print()
print(data_num_one)
print()
print(data_num_not)
print()
print(data_classify)

list = ['Product_Info_3', 'Product_Info_4', 'Ins_Age', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_2',
        'Employment_Info_4', 'Employment_Info_6', 'Medical_History_1']


def draw_categorical_variable(feature_name):
    for response_value in df.Response.unique():
        subset = df[df['Response'] == response_value]

        info_counts = subset[feature_name].value_counts().reset_index()
        info_counts.columns = [feature_name, 'count']

        plt.bar(info_counts[feature_name], info_counts['count'])

        plt.title(f'Info Distribution for Response = {response_value}')
        plt.xlabel(feature_name)
        plt.ylabel('Count')

        plt.show()

# draw_categorical_variable('Product_Info_1')
