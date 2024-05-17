# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


class DataPreprocessor:
    def __init__(self, use_cols):
        """
        初始化数据预处理器
        :param use_cols: 将使用哪些特征作为模型的训练特征
        """
        self.use_cols = use_cols

        # 原始数据中的稠密特征，即已被归一化的特征
        dense_cols = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4',
                      'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4',
                      'Family_Hist_5']
        # 只保留需要使用的特征
        self.dense_cols = [col for col in use_cols if col in dense_cols]
        self.sparse_cols = [col for col in use_cols if col not in dense_cols]
        # 只对str类的稀疏特征进行编码，其他整型的稀疏特征不处理
        self.encode_cols = ['Product_Info_2'] if 'Product_Info_2' in use_cols else []

        # todo: 特征应该如何处理？能否利用现有特征构造新的特征？
        # 对于空值，稠密特征使用中位数填充，稀疏特征使用最频繁值填充
        self.dense_imputer = SimpleImputer(strategy='median')
        self.sparse_imputers = {col: SimpleImputer(strategy='most_frequent') for col in self.sparse_cols}

        # todo: 离散化稠密特征的方式有很多种，例如按照分位数分桶，按照等距分桶等，应该如何选择？
        # 稠密特征离散化（即分桶），部分稀疏特征可采用编码方式处理
        self.kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        self.encoders = {col: LabelEncoder() for col in self.encode_cols}

    # 使用训练数据集拟合预处理器，包括填充缺失值、离散化稠密特征和编码稀疏特征
    def fit(self, df):
        # 对稠密特征的空值填充操作进行fit
        self.dense_imputer.fit(df[self.dense_cols])
        # 离散化稠密特征的fit
        self.kbins.fit(self.dense_imputer.transform(df[self.dense_cols]))

        # 对稀疏特征的空值填充操作进行fit
        for col in self.sparse_cols:
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                df[col] = df[col].astype(str)
            self.sparse_imputers[col].fit(df[[col]])
        # 对稀疏特征进行编码
        for col in self.encode_cols:
            self.encoders[col].fit(self.sparse_imputers[col].transform(df[[col]]))

    # 将预处理应用到数据集上，包括填充缺失值、离散化和编码
    def transform(self, df):
        df = df[self.use_cols].copy()  # 仅使用指定的特征

        # 离散化稠密特征
        df[self.dense_cols] = self.kbins.transform(self.dense_imputer.transform(df[self.dense_cols]))

        # 编码稀疏特征
        for col in self.sparse_cols:
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                df[col] = df[col].astype(str)
            df[[col]] = self.sparse_imputers[col].transform(df[[col]])
        for col in self.encode_cols:
            df[col] = self.encoders[col].transform(df[col])

        return df


def get_model(model_name, config=None):
    # todo: 如何选择合适的模型？模型的超参数如何确定？
    if model_name == 'random_forest':
        # return AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5),
        #                          n_estimators=args.n_estimators,
        #                          random_state=config.random_seed)
        return BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=10),
                                 n_estimators=args.n_estimators,
                                 random_state=config.random_seed)
        # 随机森林分类器
        # return RandomForestClassifier(n_estimators=args.n_estimators,
        #                               criterion='gini',
        #                               max_depth=config.max_depth,
        #                               min_samples_split=config.min_samples_split,
        #                               bootstrap=True,
        #                               random_state=config.random_seed)
    else:
        raise ValueError(f'Invalid model name: {model_name}')


def get_metric(y_true, y_pred):
    """
    获取模型性能的评价指标
    :param y_true: Ground truth (correct) target values.
    :param y_pred: Estimated targets as returned by a classifier.
    :return: 一个字典，包含模型的性能指标，键值对为指标名称和指标值
    """
    # todo: 如何评价模型的性能？有哪些指标？如何选择合适的指标？
    result = dict()
    result['report'] = classification_report(y_true, y_pred)  # 包含一些常见的分类指标
    result['accuracy'] = accuracy_score(y_true, y_pred)
    # todo: macro average与micro average的区别是什么？应该选择哪个作为最终评价指标？理由是什么？
    result['macro_f1_score'] = f1_score(y_true, y_pred, average='macro')
    result['micro_f1_score'] = f1_score(y_true, y_pred, average='micro')
    # todo: 一般多分类目标之间是无序的，例如[猫，狗，牛，羊]。但本任务的多分类目标之间是有序的，例如正确分类为2，而预估分类为1，尽管也分错了，但是比预估分类为0要好。如何在指标上表现这种有序性？
    result['mae'] = mean_absolute_error(y_true, y_pred)

    return result


def load_data(data_path):
    global preprocessor
    df = pd.read_csv(data_path)
    for column in df.columns:
        # python默认的int64类型无法表示空值，因为pandas会将其默认转换为浮点数（可表示空值）
        # 可将float类型转换为pandas.Int64类型，可以表示空值
        if pd.api.types.is_float_dtype(df[column]):
            if np.all(df[column].dropna().apply(lambda x: x == int(x))):
                df[column] = df[column].astype('Int64')
    y = df['Response'].astype('int32')  # label无需进行编码

    if preprocessor is None:
        # todo: 选择哪些特征作为训练特征？选择的策略是什么？
        # 修改use_cols，可选择需要使用的训练特征
        use_cols = ['Product_Info_2', 'Product_Info_6', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1',
                    'InsuredInfo_3', 'InsuredInfo_6', 'Insurance_History_1', 'Insurance_History_2',
                    'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8',
                    'Insurance_History_9', 'Family_Hist_1', 'Medical_History_4', 'Medical_History_6',
                    'Medical_History_9', 'Medical_History_13', 'Medical_History_16', 'Medical_History_18',
                    'Medical_History_22', 'Medical_History_23', 'Medical_History_30', 'Medical_History_33',
                    'Medical_History_39', 'Medical_History_41', 'Medical_Keyword_3', 'Medical_Keyword_15',
                    'Medical_Keyword_23', 'Medical_Keyword_25', 'Medical_Keyword_48',
                    'Product_Info_3', 'Product_Info_4', 'Ins_Age', 'Wt', 'BMI', 'Employment_Info_1',
                    'Employment_Info_2', 'Employment_Info_4', 'Employment_Info_6', 'Medical_History_1']
        preprocessor = DataPreprocessor(use_cols)
        # 【重要】在实际应用中，是无法知道test数据全局分布的，所以fit只能在train数据上进行；利用train数据的分布，对test数据进行transform
        preprocessor.fit(df)

    X = preprocessor.transform(df)
    return X, y


def train(train_data_path, model_path, config):
    X, y = load_data(train_data_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=config.random_seed)

    clf = get_model(config.model, config)
    clf.fit(X_train, y_train)

    pred_train = clf.predict(X_train)
    pred_val = clf.predict(X_val)
    train_metric = get_metric(y_true=y_train, y_pred=pred_train)
    val_metric = get_metric(y_true=y_val, y_pred=pred_val)

    print('Training Finished...')
    print('Training accuracy = {:.4}, f1_score = {:.4}, mae = {:.4f}'.format(
        train_metric['accuracy'], train_metric['micro_f1_score'], train_metric['mae']))
    print('Validation accuracy = {:.4}, f1_score = {:.4}, mae = {:.4f}'.format(
        val_metric['accuracy'], val_metric['micro_f1_score'], val_metric['mae']))

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print('Model is saved at ', os.path.join(os.getcwd(), model_path))
    return clf


def test(test_data_path, model):
    X, y = load_data(test_data_path)
    pred = model.predict(X)
    test_metric = get_metric(y_true=y, y_pred=pred)
    print('Test accuracy = {:.4}, f1_score = {:.4}, mae = {:.4f}'.format(
        test_metric['accuracy'], test_metric['micro_f1_score'], test_metric['mae']))
    print(test_metric['report'])

    print(confusion_matrix(y, pred))
    print(classification_report(y, pred))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='dataset')
    parser.add_argument('--checkpoint_dir', default='checkpoint')
    parser.add_argument('--model', default='random_forest')
    parser.add_argument('--random_seed', type=int, default=2024)
    # Random Forest 超参数
    parser.add_argument('--n_estimators', type=int, default=50)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--min_samples_split', type=int, default=20)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    np.random.seed(args.random_seed)
    train_data_path = os.path.join(args.dataset_dir, 'train.csv')
    test_data_path = os.path.join(args.dataset_dir, 'test.csv')
    model_path = os.path.join(args.checkpoint_dir, 'clf_model.pkl')

    preprocessor = None
    # todo: 如果训练耗时较长，可以修改此处逻辑为"如果模型checkpoint已经存在，则直接加载模型，否则重新训练"，方便助教评测
    model = train(train_data_path, model_path, config=args)
    test(test_data_path, model)
