import pandas as pd
import numpy as np
import os
import sys
import sklearn.datasets
from sklearn.model_selection import train_test_split


script_dir = os.path.dirname(os.path.abspath(__file__)) # Directory of the current script
data_dir = os.path.join(os.path.dirname(script_dir), 'data') # Directory of the data folder
print(f"Data directory: {data_dir}")

dataset = { 'adult': ['adult/adult.data', 'adult/adult.test'],
            'bank': 'bank+markerting/bank-additional-full.csv',
            'credit': 'Credit Card Fraud Detection/creditcard.csv',
            'higgs': 'higgs/HIGGS.csv.gz',
            'covtype': 'covertype/covtype.data.gz',
            'poker': ['poker+hand/poker-hand-training-true.data', 'poker+hand/poker-hand-testing.data'],
            'churn' : 'Telco Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',
            'wine': ['wine+quality/winequality-red.csv','wine+quality/winequality-white.csv'],
            'Santander': ['Santander Customer Transaction Prediction Dataset/train.csv', 'Santander Customer Transaction Prediction Dataset/test.csv'],
}

adult_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race',
                'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

bank_cols = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous',
    'poutcome', 'y'
]

higgs_cols = ['class', 'lepton pT', 'lepton eta', 'lepton phi', 'missing energy magnitude', 'missing energy phi',
                'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta', 'jet 2 phi', 'jet 2 b-tag',
                'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag',
                'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

covertype_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points'] + \
                ['Wilderness_Area_' + str(i) for i in range(1, 5)] + \
                ['Soil_Type_' + str(i) for i in range(1, 41)] + \
                ['Cover_Type']

poker_cols = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']

def load_data(dataset_name,seed = 999):
    if dataset_name == 'adult':
        data_train = pd.read_csv(
        os.path.join(data_dir, dataset[dataset_name][0]),
        header=None,
        names=adult_cols,
        sep=r",\s*",  # 处理逗号后的空格
        engine="python",
        na_values="?",  # 将 "?" 标记为缺失值
        )
        
        data_test = pd.read_csv(
        os.path.join(data_dir, dataset[dataset_name][1]),
        header=None,
        names=adult_cols,
        sep=r",\s*",  # 处理逗号后的空格
        engine="python",
        na_values="?",  # 将 "?" 标记为缺失值
        )
        
        data_test["income"] = data_test["income"].str.replace(".", "", regex=False)
        
        data_train = data_train.dropna()
        data_test = data_test.dropna()
        task = 'binary classification'
        print(task)
        print(data_train.shape, data_train['income'].shape)
        print(data_test.shape, data_test['income'].shape)
        return data_train, data_test
    
    elif dataset_name == 'bank':
        data = pd.read_csv(os.path.join(data_dir, dataset[dataset_name]))
        data = data.dropna()
        X,y = data.drop(columns=['y']), data['y']
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, stratify=y,shuffle=True, random_state=seed)
        task = 'binary classification'
        print(task)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train,X_test,y_train,y_test
    
    elif dataset_name == 'credit':
        data = pd.read_csv(os.path.join(data_dir, dataset[dataset_name]))
        data = data.dropna()
        X,y = data.drop(columns=['Class']), data['Class']
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, stratify=y,shuffle=True, random_state=seed)
        task = 'binary classification'
        print(task)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train,X_test,y_train,y_test
    
    elif dataset_name == 'higgs':
        data = pd.read_csv(
        os.path.join(data_dir, dataset[dataset_name]), 
        compression="gzip",     # 指定压缩格式
        header= None,            # 无表头
        names=higgs_cols,      # 指定列名
        sep=",",                # 分隔符（根据实际调整）
        dtype="float32",        # 数据类型优化（可选）
        )
        data = data.dropna()
        data = data.sample(frac=0.1, random_state=seed)  # 只使用10%的数据
        X,y = data.drop(columns=['class']), data['class']
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, stratify=y,shuffle=True, random_state=seed)
        task = 'binary classification'
        print(task)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train,X_test,y_train,y_test
    
    elif dataset_name == 'covtype':
        data = pd.read_csv(
        os.path.join(data_dir, dataset[dataset_name]),
        header=None,
        names=covertype_cols,
        sep=r",\s*",  # 处理逗号后的空格
        engine="python",
        )
        data = data.dropna()
        X,y = data.drop(columns=['Cover_Type']), data['Cover_Type']
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, stratify=y,shuffle=True, random_state=seed)
        task = 'multi-class classification'
        print(task)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train,X_test,y_train,y_test
    
    elif dataset_name == 'poker':
        data_train = pd.read_csv(
        os.path.join(data_dir, dataset[dataset_name][0]),
        header=None,
        names=poker_cols,
        sep=",",
        engine="python",
        )
        data_test = pd.read_csv(
        os.path.join(data_dir, dataset[dataset_name][1]),
        header=None,
        names=poker_cols,
        sep=",",
        engine="python",
        )
        data_train = data_train.dropna()
        data_test = data_test.dropna()
        
        X_train, y_train = data_train.drop(columns=['CLASS']), data_train['CLASS']
        X_test, y_test = data_test.drop(columns=['CLASS']), data_test['CLASS']
        task = 'multi-class classification'
        print(task)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train,X_test,y_train,y_test

    elif dataset_name == 'california':
        data = sklearn.datasets.fetch_california_housing(as_frame=True)
        data = data.frame
        data = data.dropna()
        X, y = data.drop(columns=['MedHouseVal']), data['MedHouseVal']
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2,shuffle=True, random_state=seed)
        task = 'regression'
        print(task)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train,X_test,y_train,y_test
    
    elif dataset_name == 'churn':
        data = pd.read_csv(os.path.join(data_dir, dataset[dataset_name]))
        data = data.drop(columns=['customerID'])
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data = data.dropna()
        X,y = data.drop(columns=['Churn']), data['Churn']
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, stratify=y,shuffle=True, random_state=seed)
        task = 'binary classification'
        print(task)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train,X_test,y_train,y_test
    
    elif dataset_name == 'wine':
        data1 = pd.read_csv(os.path.join(data_dir, dataset[dataset_name][0]), sep=';')
        data2 = pd.read_csv(os.path.join(data_dir, dataset[dataset_name][0]), sep=';')
        data = pd.concat([data1, data2])
        data = data.dropna()
        X,y = data.drop(columns=['quality']), data['quality']
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, stratify=y,shuffle=True, random_state=seed)
        task = 'multi-class classification'
        print(task)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train,X_test,y_train,y_test

    else:
        raise ValueError(f"Dataset {dataset_name} not found.")
    
# print(load_data('wine')[2].head())