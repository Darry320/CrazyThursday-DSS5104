import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.optim import Adam
from torch.optim import AdamW
import numpy as np
import torch
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
# Use old version of dataloader
from dataloaderversion1 import load_data

def clean_target_column(df, y_col):
    df[y_col] = df[y_col].astype(str).str.strip().str.replace('.', '', regex=False)


def run_tabnet_experiment(dataset_name):
    # === 创建结果保存目录 ===
    result_dir = f"results/{dataset_name}"
    os.makedirs(result_dir, exist_ok=True)

    # === 加载数据 ===
    train_df, test_df = load_data(dataset_name)
    y_col, task_type = {
        "adult": ("income", "classification"),
        "bank": ("y", "classification"),
        "higgs": ("class", "classification"),
        "covertype": ("Cover_Type", "classification"),
        "poker": ("CLASS", "classification"),
        "wine-red": ("quality", "regression"),
        "wine-white": ("quality", "regression"),
        "california": ("MedHouseVal", "regression"),
        "churn": ("Churn", "classification"),
        "credit": ("Class", "classification")
    }[dataset_name]
    
    # === 拆分数据与预处理 ===
    from sklearn.preprocessing import LabelEncoder

    # 处理所有非数值列：Label Encoding（联合 fit 确保 test 中不出未知值）
    print("[INFO] Encoding categorical columns if needed...")
    for col in train_df.columns:
        if col == y_col:
            continue  # 不处理目标列
        if train_df[col].dtype == 'object' or train_df[col].dtype.name == 'category':
            le = LabelEncoder()
            full_col = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
            le.fit(full_col)
            

            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))

    # 转为 numpy 并转换为 float32
    clean_target_column(train_df, y_col)
    clean_target_column(test_df, y_col)
    X_train = train_df.drop(columns=[y_col]).values.astype(np.float32)
    y_train = train_df[y_col].values
    X_test = test_df.drop(columns=[y_col]).values.astype(np.float32)
    y_test = test_df[y_col].values

    # 标准化特征
    print("[INFO] Standardizing numerical features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # === 初始化模型 ===
    if task_type == "classification":
        from pytorch_tabnet.tab_model import TabNetClassifier
        model = TabNetClassifier()
    else:
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        from pytorch_tabnet.tab_model import TabNetRegressor
        model = TabNetRegressor()

    # === 模型训练 ===
    # 100  10  1024  128
    # higgs 20 5 256 64
    
    
    """model = TabNetClassifier(
        optimizer_fn=Adam,
        optimizer_params=dict(lr=1e-2),  # 可改为 1e-2 或 5e-3
    )"""
    # adamW
    """model = TabNetClassifier(
        optimizer_fn=AdamW,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    )"""
    
    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=["accuracy" if task_type == "classification" else "rmse"],
        max_epochs=100,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
    )

    # === 保存模型 ===
    model.save_model(os.path.join(result_dir, f"tabnet_{dataset_name}"))
    

    # === 模型评估与结果保存 ===
    y_pred = model.predict(X_test)
    results = {"dataset": dataset_name, "task": task_type}


    if task_type == "classification":
        from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
        report = classification_report(y_test, y_pred, output_dict=True)
        results.update({
            "accuracy": report["accuracy"],
            "macro avg": report["macro avg"],
            "weighted avg": report["weighted avg"]
        })

        # 保存分类报告
        with open(os.path.join(result_dir, "classification_report.json"), "w") as f:
            json.dump(report, f, indent=4)

        # 可选：绘图也可以保存
        # === 混淆矩阵绘图与保存 ===
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title(f"{dataset_name} - Confusion Matrix")
        plt.tight_layout()
        # plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
        plt.close()

    elif task_type == "regression":
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # 代替 squared=False
        mae = mean_absolute_error(y_test, y_pred)
        # === 标准化 RMSE ===
        y_range = np.max(y_test) - np.min(y_test)
        nrmse = rmse / y_range if y_range != 0 else rmse  # 防止除零错误

        results.update({
            "rmse": rmse,
            "nrmse": nrmse,
            "mae": mae
        })

        with open(os.path.join(result_dir, "regression_results.json"), "w") as f:
            json.dump(results, f, indent=4)

    print(f"[✔] Results and model saved to: {result_dir}")

# run_tabnet_experiment("adult")
