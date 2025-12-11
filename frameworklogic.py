import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical


class DataManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.df_all = None

    def load_cicids_files(self, csv_files):
        dfs = []
        for f in csv_files:
            path = os.path.join(self.base_path, f)
            print(f"[DataManager] Loading {path}")
            df_part = pd.read_csv(path)
            df_part["SourceFile"] = f
            dfs.append(df_part)
        self.df_all = pd.concat(dfs, ignore_index=True)
        self.df_all.columns = self.df_all.columns.str.strip()
        print("[DataManager] Loaded:", self.df_all.shape)
        return self.df_all

    def split_by_day(self, day_patterns):
        subsets = {}
        for key, patterns in day_patterns.items():
            mask = False
            for p in patterns:
                mask = mask | self.df_all["SourceFile"].str.contains(p, case=False, regex=True)
            subsets[key] = self.df_all[mask].copy()
            print(f"[DataManager] Subset '{key}' shape:", subsets[key].shape)
        return subsets

    @staticmethod
    def balanced_sample(df, label_col="LabelEnc", n_per_class=200000, random_state=42):
        groups = []
        for lab, g in df.groupby(label_col):
            take = min(len(g), n_per_class)
            groups.append(g.sample(n=take, random_state=random_state))
        df_bal = pd.concat(groups, ignore_index=True)
        print(f"[DataManager] Balanced sample shape: {df_bal.shape}")
        return df_bal


class Preprocessor:
    def __init__(self):
        self.drop_cols = [
            'Timestamp', 'Flow ID', 'Src IP', 'Dst IP',
            'Src Port', 'Dst Port', 'Protocol'
        ]
        self.scaler = None
        self.label_encoder = None
        self.num_cols = None

    def initial_clean(self, df: pd.DataFrame):
        df = df.copy()
        for col in self.drop_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        df = df.dropna(subset=["Label"])
        return df

    def fit_label_encoder(self, df_all: pd.DataFrame):
        self.label_encoder = LabelEncoder()
        df_all["LabelEnc"] = self.label_encoder.fit_transform(df_all["Label"])
        print("[Preprocessor] Label classes:", list(self.label_encoder.classes_))
        return df_all

    def prepare_features(self, df: pd.DataFrame, label_col="LabelEnc"):
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.num_cols = num_cols

        X = df[num_cols].copy()
        if label_col in X.columns:
            X.drop(columns=[label_col], inplace=True)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.fillna(X.median(numeric_only=True))
        X = X.astype("float32")

        y = df[label_col].to_numpy()
        return X, y

    def fit_scaler(self, X_train):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        print("[Preprocessor] Scaler fitted on shape:", X_train.shape)

    def transform(self, X):
        return self.scaler.transform(X)

    def encode_labels(self, y):
        num_classes = len(self.label_encoder.classes_)
        return to_categorical(y, num_classes=num_classes)

    def save_preprocessing(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(dir_path, "scaler.joblib"))
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, os.path.join(dir_path, "label_encoder.joblib"))
        if self.num_cols is not None:
            with open(os.path.join(dir_path, "num_cols.json"), "w", encoding="utf-8") as f:
                json.dump(self.num_cols, f, ensure_ascii=False, indent=2)
        print(f"[Preprocessor] Saved preprocessing artifacts to {dir_path}")

    def load_preprocessing(self, dir_path: str):
        self.scaler = joblib.load(os.path.join(dir_path, "scaler.joblib"))
        self.label_encoder = joblib.load(os.path.join(dir_path, "label_encoder.joblib"))
        num_cols_path = os.path.join(dir_path, "num_cols.json")
        if os.path.exists(num_cols_path):
            with open(num_cols_path, "r", encoding="utf-8") as f:
                self.num_cols = json.load(f)
        print(f"[Preprocessor] Loaded preprocessing artifacts from {dir_path}")


class BPNNModelManager:
    def __init__(self, input_dim: int, num_classes: int, model=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        if model is not None:
            self.model = model
        else:
            self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("[BPNN] Model built.")
        return model

    def train(self, X_train, y_train_cat, epochs=15, batch_size=512, validation_split=0.2):
        history = self.model.fit(
            X_train,
            y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history

    def predict(self, X):
        proba = self.model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)

    def evaluate(self, X, y_true, label_encoder: LabelEncoder, subset_name=""):
        y_pred = self.predict(X)
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        labels_present = np.unique(y_true)
        target_names = label_encoder.inverse_transform(labels_present)
        report = classification_report(
            y_true, y_pred,
            labels=labels_present,
            target_names=target_names,
            zero_division=0
        )

        print(f"\n[Evaluate] {subset_name}")
        print("Accuracy:", acc)
        print("F1-macro:", f1_macro)
        print("F1-weighted:", f1_weighted)
        print(report)

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "report": report
        }

    def save(self, dir_path: str, version_name: str = "v1"):
        os.makedirs(dir_path, exist_ok=True)
        model_path = os.path.join(dir_path, f"bpnn_{version_name}.h5")
        self.model.save(model_path)
        meta = {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "version": version_name
        }
        with open(os.path.join(dir_path, f"bpnn_{version_name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[BPNN] Saved model + metadata to {dir_path}")

    @staticmethod
    def load(dir_path: str, version_name: str = "v1"):
        meta_path = os.path.join(dir_path, f"bpnn_{version_name}_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        input_dim = meta["input_dim"]
        num_classes = meta["num_classes"]

        model_path = os.path.join(dir_path, f"bpnn_{version_name}.h5")
        model = load_model(model_path)

        print(f"[BPNN] Loaded model '{version_name}' from {dir_path}")
        return BPNNModelManager(input_dim=input_dim, num_classes=num_classes, model=model)


class AdaptiveMonitoringFramework:
    def __init__(self, base_path, csv_files):
        self.data_manager = DataManager(base_path)
        self.prep = Preprocessor()
        self.csv_files = csv_files

        self.label_encoder = None
        self.model_manager = None

        self.df_mon_thu = None
        self.df_fri = None

    def load_and_prepare(self):
        df_all = self.data_manager.load_cicids_files(self.csv_files)
        df_all = self.prep.initial_clean(df_all)
        df_all = self.prep.fit_label_encoder(df_all)

        self.data_manager.df_all = df_all
        self.label_encoder = self.prep.label_encoder

        subsets = self.data_manager.split_by_day({
            "mon_thu": ["Monday", "Tuesday", "Wednesday", "Thursday"],
            "fri": ["Friday"]
        })
        self.df_mon_thu = subsets["mon_thu"]
        self.df_fri = subsets["fri"]

    def run_phase1_baseline(self, n_per_class_mon_thu=20000, n_per_class_fri=20000):
        df_mon_thu_bal = self.data_manager.balanced_sample(
            self.df_mon_thu, label_col="LabelEnc",
            n_per_class=n_per_class_mon_thu
        )
        df_fri_bal = self.data_manager.balanced_sample(
            self.df_fri, label_col="LabelEnc",
            n_per_class=n_per_class_fri
        )

        X_mon_thu, y_mon_thu = self.prep.prepare_features(df_mon_thu_bal, "LabelEnc")
        X_fri, y_fri = self.prep.prepare_features(df_fri_bal, "LabelEnc")

        self.prep.fit_scaler(X_mon_thu)
        X_mon_thu_scaled = self.prep.transform(X_mon_thu)
        X_fri_scaled = self.prep.transform(X_fri)

        input_dim = X_mon_thu_scaled.shape[1]
        num_classes = len(self.label_encoder.classes_)
        self.model_manager = BPNNModelManager(input_dim, num_classes)

        y_mon_thu_cat = self.prep.encode_labels(y_mon_thu)
        self.model_manager.train(X_mon_thu_scaled, y_mon_thu_cat)

        metrics_p1 = self.model_manager.evaluate(
            X_fri_scaled, y_fri,
            self.label_encoder,
            subset_name="[Phase 1] Friday"
        )
        return metrics_p1, (X_mon_thu_scaled, y_mon_thu, X_fri_scaled, y_fri)

    def run_phase2_adaptation(self, X_mon_thu_scaled, y_mon_thu, X_fri_scaled, y_fri, test_size=0.3):
        X_fri_train, X_fri_test, y_fri_train, y_fri_test = train_test_split(
            X_fri_scaled, y_fri,
            test_size=test_size,
            random_state=42,
            stratify=y_fri
        )

        X_train2 = np.vstack([X_mon_thu_scaled, X_fri_train])
        y_train2 = np.concatenate([y_mon_thu, y_fri_train])
        y_train2_cat = self.prep.encode_labels(y_train2)

        input_dim = X_train2.shape[1]
        num_classes = len(self.label_encoder.classes_)
        self.model_manager = BPNNModelManager(input_dim, num_classes)

        self.model_manager.train(X_train2, y_train2_cat)

        metrics_p2 = self.model_manager.evaluate(
            X_fri_test, y_fri_test,
            self.label_encoder,
            subset_name="[Phase 2] Friday-test"
        )
        return metrics_p2

    def save_current_state(self, dir_path: str, version_name: str):
        if self.model_manager is None:
            print("[Framework] No model to save!")
            return
        self.model_manager.save(dir_path, version_name=version_name)
        self.prep.save_preprocessing(dir_path)
        print(f"[Framework] Saved full state as '{version_name}'")

    def load_state(self, dir_path: str, version_name: str):
        self.prep.load_preprocessing(dir_path)
        self.label_encoder = self.prep.label_encoder
        self.model_manager = BPNNModelManager.load(dir_path, version_name=version_name)
        print(f"[Framework] Loaded state '{version_name}'")
