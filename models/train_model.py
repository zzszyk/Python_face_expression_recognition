import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.ensemble import BaggingClassifier
import joblib
import os


def main():
    # 加载特征和标签
    features_path = r'D:\code\Python_Mediapipe\face_expression_recognition\data\features.npy'
    labels_path = r'D:\code\Python_Mediapipe\face_expression_recognition\data\labels.npy'

    try:
        features = np.load(features_path)
        labels = np.load(labels_path)
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    # 特征标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    # 使用 GridSearchCV 进行模型调优
    param_grid = {
        'estimator__C': [0.01, 0.1, 1, 10, 100, 1000],
        'estimator__kernel': ['linear', 'rbf', 'poly'],
        'estimator__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'estimator__degree': [2, 3, 4, 5]
    }
    model = svm.SVC()
    bagging_model = BaggingClassifier(estimator=model, n_estimators=10, random_state=42)
    grid_search = GridSearchCV(bagging_model, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # 使用最佳参数训练模型
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # 评估模型
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f'准确率：{accuracy}')
    print(f'召回率：{recall}')
    print(f'F1分数：{f1}')

    # 保存模型
    model_path = os.path.join(os.path.dirname(__file__),
                              r'D:\code\Python_Mediapipe\face_expression_recognition\models\expression_recognition_model.pkl')
    try:
        joblib.dump(best_model, model_path)
        print(f"模型已保存到: {os.path.abspath(model_path)}")

        # 保存标准化器
        scaler_path = os.path.join(os.path.dirname(__file__),
                                   r'D:\code\Python_Mediapipe\face_expression_recognition\models\scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"标准化器已保存到: {os.path.abspath(scaler_path)}")
    except Exception as e:
        print(f"保存模型或标准化器时出错: {e}")


if __name__ == '__main__':
    main()