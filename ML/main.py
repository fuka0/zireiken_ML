from pathlib import Path
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical, plot_model
from keras.models import load_model
from module3.create_detaset import select_electrode, load_data, make_columns, generate_noise
from module2.create_model import one_dim_CNN_model
from sklearn.model_selection import StratifiedShuffleSplit
import os
from sklearn.metrics import confusion_matrix
import random
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["OMP_NUM_THREADS"] = "1"


def generate_string(decompose_level, d_num):
    # decompose_levelから開始して、d_numの数だけ逆順にdetailを取得
    details = [f"d{i}" for i in range(decompose_level, decompose_level - d_num, -1)]
    # 文字列を"_"で結合
    return "_".join(details)

def Preprocessing(preprocessing_type):
    if preprocessing_type == "d":
        preprocessing_dir = "DWT_data"
    elif preprocessing_type == "e":
        preprocessing_dir = "Envelope_data"
    elif preprocessing_type == "b" :
        preprocessing_dir = "BPF_data"
    return preprocessing_dir
# ////////////////////////////////////////////////////////////////////////////////////////
n_class = 2 # 何クラス分類か(左手と右手なら2, 両手も含むなら3, 両足も含むなら4)
number_of_chs =[64]
movement_types = ["left_right_fist", "fists_feet"] # 運動タイプを定義

extraction_section = True # 切り出し区間が安静時を含まないならTrue,含むならFalse
baseline_correction = True # ベースライン補正の有無
ext_sec = "move_only" if extraction_section else "rest_move"
baseline = "baseline_true" if baseline_correction else "baseline_false"
preprocessing_type= "d" # d(DWT), e(Envelope), b(BPF)
ds = 2 # ダウンサンプリングの設定

d_num = 3 # 取得するdetailの個数(上から順に{D4,D3...})(2 or 3)
decompose_level = 5 # 分解レベル


reduce_data = False # データ削減を有効にするか
num_samples = 90  # 選択するサンプル数

sss = StratifiedShuffleSplit(n_splits=5,random_state=17,test_size=0.2)
sss_val = StratifiedShuffleSplit(n_splits=1,random_state=17,test_size=0.2)

for number_of_ch in number_of_chs:
    ch_idx, extracted_ch = select_electrode(number_of_ch)
    current_dir = Path.cwd()
    preprocessing_dir = Preprocessing(preprocessing_type)

    columns = make_columns(n_class)
    subjects_dir = [current_dir/preprocessing_dir/ext_sec/baseline/f"S{i+1:03}" for i in range(104)]
    subjects_name = [subject_dir.name for subject_dir in subjects_dir]
    random.seed(22)
    target_subjects = random.sample(subjects_name, 5)

    details_dir = generate_string(decompose_level, d_num)

    for target_subject in target_subjects:
        if target_subject == "S032" or target_subject == "S004":
            target_subject_index = target_subjects.index(target_subject)
            df = pd.DataFrame(index=[f"target_{target_subject}", "ACC", "PRE", "REC", "F1"], columns=columns)

            if preprocessing_dir == "DWT_data":
                data_paths = list((current_dir / "ML" / "ref_data" / preprocessing_dir / f"decomposition_level{decompose_level}" / details_dir / ext_sec / f"ds_{ds}" ).glob(f"S*/*.npy"))
            elif preprocessing_dir == "Envelope_data":
                data_paths = list((current_dir / "ML" / "ref_data" / preprocessing_dir / ext_sec / f"ds_{ds}" ).glob(f"S*/*.npy"))
            elif preprocessing_dir == "BPF_data":
                data_paths = list((current_dir / "ML" / "ref_data" / preprocessing_dir / ext_sec / f"ds_{ds}" ).glob(f"S*/*.npy"))

            for data_path in data_paths:
                if data_path.parent.name == target_subject:
                    # ターゲットとなる被験者のデータpath
                    target_subject_data_path = data_path
                    # data_pathからtarget_subjectのpathを除外
                    data_paths.remove(data_path)

            X, y = load_data(data_paths, movement_types, ch_idx, n_class, number_of_ch)
            left_fist_idx = np.where(y == 0)
            right_fist_idx = np.where(y == 1)
            both_fists = np.where(y == 2)
            both_feet = np.where(y == 3)

            # Lists to store the accuracy of each fold
            train_acc_list = []
            val_acc_list = []

            # 混同行列を格納するためのリスト
            conf_matrices = []
            # recall, precision, F-valueを格納するためのリスト
            recalls, precisions, f_values = [], [], []

            # Split the dataset into training and testing sets for each fold
            for train, test in sss.split(X, y):
                X_train, X_test = X[train], X[test]
                y_train, y_test = y[train], y[test]
                for train_idx, val in sss_val.split(X_train, y_train):
                    X_train, X_val = X_train[train_idx], X_train[val]
                    y_train, y_val = y_train[train_idx], y_train[val]

                # ノイズ生成
                noise = generate_noise(X_train, "gauss")
                X_train_noised = X_train + noise

                # Convert labels to categorical
                y_train = to_categorical(y_train, num_classes=n_class)
                y_test = to_categorical(y_test, num_classes=n_class)
                y_val = to_categorical(y_val, num_classes=n_class)

                X_train_combined = np.concatenate([X_train, X_train_noised])
                y_train_combined = np.concatenate([y_train, y_train])

                batch_size = X_train_combined.shape[0]
                timesteps = X_train_combined.shape[1]
                input_dim = X_train_combined.shape[2]
                input_shape = (X_train_combined.shape[1], X_train_combined.shape[2])
                # modelを定義
                model = one_dim_CNN_model(input_shape, n_class, optimizer='adam', learning_rate=0.001)
                # plot_model(model, to_file="global_model.png", show_shapes=True)
                if preprocessing_dir == "DWT_data":
                    log_dir = current_dir / "ML" / "logs" / preprocessing_dir.split("_")[0] / f"decomposition_level{decompose_level}" / details_dir / ext_sec / f"{number_of_ch}ch" / f"ds_{ds}" / f"{n_class}class"
                    model_dir = "ML" / Path("model_container")/preprocessing_dir.split('_')[0]/f"decomposition_level{decompose_level}" / details_dir /ext_sec/f"{number_of_ch}ch"/f"ds_{ds}"/f"{n_class}class"
                    os.makedirs(log_dir, exist_ok=True)
                    os.makedirs(model_dir, exist_ok=True)
                elif preprocessing_dir == "Envelope_data" or preprocessing_dir == "BPF_data":
                    log_dir = current_dir / "ML" / "logs" / preprocessing_dir.split("_")[0] / ext_sec / f"{number_of_ch}ch" / f"ds_{ds}" / f"{n_class}class"
                    model_dir = "ML" / Path("model_container")/preprocessing_dir.split('_')[0] / ext_sec/f"{number_of_ch}ch"/f"ds_{ds}"/f"{n_class}class"
                    os.makedirs(log_dir, exist_ok=True)
                    os.makedirs(model_dir, exist_ok=True)

                callbacks_list = [EarlyStopping(monitor="val_loss", patience=4),
                                ModelCheckpoint(f"{model_dir}/target_{target_subject}_model.keras", save_best_only=True, monitor='val_loss')]

                history = model.fit(X_train_combined, y_train_combined, epochs=200, batch_size=12, validation_data=(X_val, y_val), callbacks=callbacks_list)
        # 転移学習part//////////////////////////////////
            group_results = []
            sstl_train_acc_list = []
            sstl_val_acc_list = []

            # データのロード
            X_sstl, y_sstl = load_data([target_subject_data_path], movement_types, ch_idx, n_class, number_of_ch)
            # ラベルをカテゴリカルに変換
            y_sstl = to_categorical(y_sstl, num_classes=n_class)

            if reduce_data:
                # ランダムにデータを選択するためのインデックスを生成
                indices = np.arange(X_sstl.shape[0])
                np.random.shuffle(indices)
                indices = indices[:num_samples]

                # 選択したインデックスを使用してデータを抽出
                X_sstl = X_sstl[indices]
                y_sstl = y_sstl[indices]
            else:
                pass

            # 混同行列を格納するためのリスト
            conf_matrices = []
            # recall, precision, F-valueを格納するためのリスト
            recalls, precisions, f_values = [], [], []
            # 各分割での訓練と検証
            for train, test in sss.split(X_sstl, y_sstl):
                # グローバルモデルのロード
                global_model = load_model(f"{model_dir}/target_{target_subject}_model.keras")
                for layer in global_model.layers:
                    if layer.name == "L4":
                        layer.trainable = True
                        break
                    layer.trainable = False

                # # 確認のためにモデルのtrainable属性を出力
                # for layer in global_model.layers:
                #     print(layer.name, layer.trainable)
                # plot_model(global_model, to_file="SS-TL_model.png", show_shapes=True)
                X_train_sstl, X_test_sstl = X_sstl[train], X_sstl[test]
                y_train_sstl, y_test_sstl = y_sstl[train], y_sstl[test]


                sstl_callbacks_list = [EarlyStopping(monitor="val_loss", patience=4)]

                # SS-TL学習
                history_sstl = global_model.fit(X_train_sstl, y_train_sstl, epochs=30, batch_size=12, validation_data=(X_test_sstl, y_test_sstl), callbacks=sstl_callbacks_list)
                y_pred = global_model.predict(X_test_sstl)

                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_test_sstl, axis=1)

                # 混同行列を計算
                conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
                conf_matrices.append(conf_matrix)

            # 混同行列の平均を計算
            average_conf_matrix = np.mean(conf_matrices, axis=0)

# 混同行列のヒートマップをプロット
plt.figure(figsize=(8, 6))
sns.heatmap(average_conf_matrix, annot=True, cmap="Blues", fmt=".1f",
            xticklabels=["Left Fist", "Right Fist"],
            yticklabels=["Left Fist", "Right Fist"])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Average Confusion Matrix Heatmap')
plt.show()
