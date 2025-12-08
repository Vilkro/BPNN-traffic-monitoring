# import os
# import pandas as pd
# import numpy as np
#
# base_path = "D:/TrafficClassification/MachineLearningCVE"
#
# csv_files = [
#     "Monday-WorkingHours.pcap_ISCX.csv",
#     "Tuesday-WorkingHours.pcap_ISCX.csv",
#     "Wednesday-workingHours.pcap_ISCX.csv",
#     "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
#     "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
#     "Friday-WorkingHours-Morning.pcap_ISCX.csv",
#     "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
#     "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
# ]
#
# dfs = []
# for f in csv_files:
#     df_part = pd.read_csv(os.path.join(base_path, f))
#     df_part["SourceFile"] = f
#     dfs.append(df_part)
#
# df_all = pd.concat(dfs, ignore_index=True)
# df_all.columns = df_all.columns.str.strip()
#
# print("Full dataset shape:", df_all.shape)
#
# # drop id columns
# drop_cols = [
#     'Timestamp', 'Flow ID', 'Src IP', 'Dst IP',
#     'Src Port', 'Dst Port', 'Protocol'
# ]
# for col in drop_cols:
#     if col in df_all.columns:
#         df_all.drop(columns=[col], inplace=True)
#
# # remove rows without label
# df_all = df_all.dropna(subset=["Label"])
#
# mask_mon_thu = df_all["SourceFile"].str.contains("Monday|Tuesday|Wednesday|Thursday", case=False, regex=True)
# mask_fri     = df_all["SourceFile"].str.contains("Friday", case=False, regex=True)
#
# df_mon_thu = df_all[mask_mon_thu].copy()
# df_fri     = df_all[mask_fri].copy()
#
# print("Mon–Thu raw shape:", df_mon_thu.shape)
# print("Friday raw shape:", df_fri.shape)
#
# from sklearn.preprocessing import LabelEncoder
#
# # encode labels
# le = LabelEncoder()
# df_all["LabelEnc"] = le.fit_transform(df_all["Label"])
# print("Number of classes total:", len(le.classes_))
#
# # attach encoded labels to subsets
# df_mon_thu["LabelEnc"] = df_all.loc[df_mon_thu.index, "LabelEnc"]
# df_fri["LabelEnc"]     = df_all.loc[df_fri.index, "LabelEnc"]
#
# # balanced sampling
# def balanced_sample(df, label_col, n_per_class=100000, random_state=42):
#     groups = []
#     for lab, g in df.groupby(label_col):
#         take = min(len(g), n_per_class)
#         groups.append(g.sample(n=take, random_state=random_state))
#     return pd.concat(groups, ignore_index=True)
#
# df_mon_thu_bal = balanced_sample(df_mon_thu, "LabelEnc", n_per_class=100000)
# df_fri_bal     = balanced_sample(df_fri, "LabelEnc", n_per_class=100000)
#
# print("Mon–Thu balanced shape:", df_mon_thu_bal.shape)
# print("Friday balanced shape:", df_fri_bal.shape)
# print("Mon–Thu class counts:\n", df_mon_thu_bal["LabelEnc"].value_counts())
# print("Friday class counts:\n", df_fri_bal["LabelEnc"].value_counts())
#
# num_cols = df_mon_thu_bal.select_dtypes(include=[np.number]).columns
#
#
# def prepare_X_y(df, num_cols, label_col="LabelEnc"):
#     X = df[num_cols].copy()
#
#     if label_col in X.columns:
#         X = X.drop(columns=[label_col])
#
#     X.replace([np.inf, -np.inf], np.nan, inplace=True)
#     X = X.fillna(X.median(numeric_only=True))
#
#     X = X.astype("float32")
#     y = df[label_col].to_numpy()
#     return X, y
#
#
# X_mon_thu, y_mon_thu = prepare_X_y(df_mon_thu_bal, num_cols)
# X_fri, y_fri = prepare_X_y(df_fri_bal, num_cols)
#
# print("Mon–Thu X shape:", X_mon_thu.shape)
# print("Friday X shape:", X_fri.shape)
#
# import numpy as np
#
# print("Any NaN Mon–Thu:", np.isnan(X_mon_thu.to_numpy()).any())
# print("Any NaN Friday :", np.isnan(X_fri.to_numpy()).any())
# print("Any inf Mon–Thu:", np.isinf(X_mon_thu.to_numpy()).any())
# print("Any inf Friday :", np.isinf(X_fri.to_numpy()).any())
#
# from sklearn.model_selection import train_test_split
#
# X_fri_train, X_fri_test, y_fri_train, y_fri_test = train_test_split(
#     X_fri,
#     y_fri,
#     test_size=0.3,
#     random_state=42,
#     stratify=y_fri
# )
#
# print("Friday train:", X_fri_train.shape, " Friday test:", X_fri_test.shape)
#
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# scaler.fit(X_mon_thu)
#
# X_mon_thu_scaled   = scaler.transform(X_mon_thu)
# X_fri_scaled       = scaler.transform(X_fri)
# X_fri_train_scaled = scaler.transform(X_fri_train)
# X_fri_test_scaled  = scaler.transform(X_fri_test)
#
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.utils import to_categorical
#
# num_classes = len(le.classes_)
# input_dim = X_mon_thu_scaled.shape[1]
# print("Input dim:", input_dim, "Num classes:", num_classes)
#
# def build_bpnn_stronger(input_dim, num_classes):
#     model = Sequential()
#     model.add(Input(shape=(input_dim,)))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.4))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model
#
# from sklearn.metrics import classification_report, accuracy_score, f1_score
#
# # Phase 1 training
# y_mon_thu_cat = to_categorical(y_mon_thu, num_classes=num_classes)
#
# model_p1 = build_bpnn_stronger(input_dim, num_classes)
#
# history_p1 = model_p1.fit(
#     X_mon_thu_scaled,
#     y_mon_thu_cat,
#     epochs=15,             # moderate training
#     batch_size=512,        # smaller batch to reduce RAM spikes
#     validation_split=0.2,
#     verbose=1
# )
#
# # Phase 1 evaluation on all Friday
# y_fri_pred_proba_p1 = model_p1.predict(X_fri_scaled, verbose=0)
# y_fri_pred_p1 = np.argmax(y_fri_pred_proba_p1, axis=1)
#
# print("\n[Phase 1] Friday evaluation (balanced subset):")
# print("Accuracy:", accuracy_score(y_fri, y_fri_pred_p1))
# print("F1-macro   :", f1_score(y_fri, y_fri_pred_p1, average="macro"))
# print("F1-weighted:", f1_score(y_fri, y_fri_pred_p1, average="weighted"))
#
# labels_fri = np.unique(y_fri)
# target_names_fri = le.inverse_transform(labels_fri)
#
# print("\n[Phase 1] Classification report (Friday):")
# print(classification_report(
#     y_fri,
#     y_fri_pred_p1,
#     labels=labels_fri,
#     target_names=target_names_fri,
#     zero_division=0
# ))
#
# print("\n=== Phase 2: Adaptation with Mon–Thu + Friday-train ===")
#
# # combine Mon–Thu and Friday-train for adaptation
# X_phase2_train = np.vstack([X_mon_thu_scaled, X_fri_train_scaled])
# y_phase2_train = np.concatenate([y_mon_thu, y_fri_train])
# y_phase2_cat   = to_categorical(y_phase2_train, num_classes=num_classes)
#
# model_p2 = build_bpnn_stronger(input_dim, num_classes)
#
# history_p2 = model_p2.fit(
#     X_phase2_train,
#     y_phase2_cat,
#     epochs=15,
#     batch_size=512,
#     validation_split=0.2,
#     verbose=1
# )
#
# # evaluate on Friday-test
# y_fri_test_pred_proba = model_p2.predict(X_fri_test_scaled, verbose=0)
# y_fri_test_pred = np.argmax(y_fri_test_pred_proba, axis=1)
#
# print("\n[Phase 2] Friday-test evaluation (after adaptation):")
# print("Accuracy:", accuracy_score(y_fri_test, y_fri_test_pred))
# print("F1-macro   :", f1_score(y_fri_test, y_fri_test_pred, average="macro"))
# print("F1-weighted:", f1_score(y_fri_test, y_fri_test_pred, average="weighted"))
#
# labels_fri_test = np.unique(y_fri_test)
# target_names_fri_test = le.inverse_transform(labels_fri_test)
#
# print("\n[Phase 2] Classification report (Friday-test):")
# print(classification_report(
#     y_fri_test,
#     y_fri_test_pred,
#     labels=labels_fri_test,
#     target_names=target_names_fri_test,
#     zero_division=0
# ))