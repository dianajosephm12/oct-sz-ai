from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import utils
from sklearn.decomposition import PCA
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import oct_data.fep as fep
import oct_data.fep_control as fep_control
import oct_data.chronic_control as chronic_control
import oct_data.chronic as chronic
import oct_data.fep_clinical as fep_clinical
import oct_data.fep_control_clinical as fep_control_clinical
import oct_data.chronic_clinical as chronic_clinical
import oct_data.chronic_control_clinical as chronic_control_clinical

# GET ACTIVATIONS
# dir1 = 'Control30Up_intermediates'  # 0
# dir2 = 'Chronic_intermediates'   # 1
dir1 = 'ControlUnder30_intermediates'  # 0
dir2 = 'FEP_intermediates'   # 1
activation = 'e3'
is_e2_e3 = True     # True = needs different class weights for control vs patient omitted
control_files = [f for f in os.listdir(dir1) if f.startswith('{activation}_'.format(activation=activation))]
patient_files = [f for f in os.listdir(dir2) if f.startswith('{activation}_'.format(activation=activation))]

l_control = len(control_files)
l_pt = len(patient_files)
control_labels = [0] * l_control
pt_labels = [1] * len(patient_files)
labels = control_labels + pt_labels     # labels for input_array items

control_data = []
pt_data = []
for f in control_files:
    fl_npy = np.load('{dir}/{file}'.format(dir=dir1, file=f)).flatten()
    control_data.append(fl_npy)
for f in patient_files:
    fl_npy = np.load('{dir}/{file}'.format(dir=dir2, file=f)).flatten()
    pt_data.append(fl_npy)
input_array = control_data + pt_data    # array of shape (vector, num_features)
# print(len(input_array))

# USE ALL OCT DATA ONLY
# control_data = fep_control.fep_control_oct
# pt_data = fep.fep_oct
# control_data = chronic_control.chronic_control
# pt_data = chronic.chronic
# USE OCT + CLINICAL DATA
# pt_data = fep_clinical.fep_full_clinical
# control_data = fep_control_clinical.fep_control_full_clinical
# pt_data = chronic_clinical.chronic_full_clinical
# control_data = chronic_control_clinical.chronic_control_full_clinical
# USE CLINICAL DATA ONLY
# control_data = chronic_control_clinical.chronic_control_clinical_only
# pt_data = chronic_clinical.chronic_clinical_only
# control_data = fep_control_clinical.fep_control_clinical_only
# pt_data = fep_clinical.fep_clinical_only
# USE OCT STUDY OCT DATA ONLY
# control_data = fep_control.fep_control_study_oct
# pt_data = fep.fep_study_oct
# control_data = chronic_control.chronic_control_study_oct
# pt_data = chronic.chronic_study_oct
# USE OCT VOLUME AND CSF ONLY
# pt_data = fep.fep_used
# control_data = fep_control.fep_control_used
# pt_data = chronic.chronic_used
# control_data = chronic_control.chronic_control_used

# new_control = []
# new_pt = []
# for c in control_data:
#     csf = (c[0] + c[2]) / 2.0
#     vc = (c[1] + c[3]) / 2.0
#     new_control.append([csf, vc])
# for p in pt_data:
#     csf = (p[0] + p[2]) / 2.0
#     vc = (p[1] + p[3]) / 2.0
#     new_pt.append([csf, vc])
# control_data = new_control
# pt_data = new_pt


# l_control = len(control_data)
# l_pt = len(pt_data)
# control_labels = [0] * l_control
# pt_labels = [1] * l_pt
# labels = control_labels + pt_labels
# input_array = control_data + pt_data

# MAKE CLASSIFIERS WITH REMOVED SAMPLES AND TEST
# true_pos = 0
# true_neg = 0
# false_pos = 0
# false_neg = 0
# m = 1.0
# x_weight = 34.0/(40*m)
# y_weight = 34.0/(28*m)
# # x_weight = .7368421
# # y_weight = 1.0
# weights = {0: x_weight, 1: y_weight}
# print(weights)

def get_weights(m, num_control, num_pt):
    x_weight = (num_control + num_pt) / (num_control * 2 * m)
    y_weight = (num_control + num_pt) / (num_pt * 2 * m)
    return x_weight, y_weight

def get_metrics(labels, preds):
    accuracy = metrics.accuracy_score(labels, preds)
    precision = metrics.precision_score(labels, preds)
    recall = metrics.recall_score(labels, preds)
    f1 = (2 * precision * recall) / (precision + recall)
    return  accuracy, precision, recall, f1


def build_linear_svc(input_array):
    oct_features = ['OS_RNFL', 'OS_Cup_Disc_Ratio', 'OS_Cup_Volume', 'OD_RNFL', 'OD_Cup_Disc_Ratio', 'OD_Cup_Volume', 'RNFLsymmetry', 'OS_SUP_value', 'OS_TEM_value', 'OS_INF_value', 'OS_NAS_value', 'OD_SUP_value', 'OD_TEM_value', 'OD_INF_value', 'OD_NAS_value', 'LMac12', 'LMac3', 'LMac6', 'LMac9', 'RMac12', 'RMac3', 'RMac6', 'RMac9', 'OS_macula_csf', 'OS_Volume_Cube', 'OS_Thickness_Avg_Cube', 'OD_macula_csf', 'OD_Volume_Cube', 'OD_Thickness_Avg_Cube', 'GCL_IPL_Avg_OS', 'GCL_IPL_Avg_OD']
    fep_oct_clinical_features = ['Sex', 'Race', 'Simp_Race', 'Ethnicity', 'Handedness', 'Age', 'WTAR_raw', 'WTAR_scaled', 'Corrected_vision', 'OSLogMar', 'ODLogMar', 'BinocularLogMar', 'Education', 'Degree', 'OS_RNFL', 'OS_Cup_Disc_Ratio', 'OS_Cup_Volume', 'OD_RNFL', 'OD_Cup_Disc_Ratio', 'OD_Cup_Volume', 'RNFLsymmetry', 'OS_SUP_value', 'OS_TEM_value', 'OS_INF_value', 'OS_NAS_value', 'OD_SUP_value', 'OD_TEM_value', 'OD_INF_value', 'OD_NAS_value', 'LMac12', 'LMac3', 'LMac6', 'LMac9', 'RMac12', 'RMac3', 'RMac6', 'RMac9', 'OS_macula_csf', 'OS_Volume_Cube', 'OS_Thickness_Avg_Cube', 'OD_macula_csf', 'OD_Volume_Cube', 'OD_Thickness_Avg_Cube', 'GCL_IPL_Avg_OS', 'GCL_IPL_Avg_OD']
    chronic_oct_clinical_features = ['Age', 'Sex', 'Race', 'Simp_Race', 'Ethnicity', 'Handedness', 'WTAR_raw', 'WTAR_scaled', 'Corrected_vision', 'OSLogMar', 'ODLogMar', 'BinocularLogMar', 'Education', 'Degree', 'OS_RNFL', 'OS_Cup_Disc_Ratio', 'OS_Cup_Volume', 'OD_RNFL', 'OD_Cup_Disc_Ratio', 'OD_Cup_Volume', 'RNFLsymmetry', 'OS_SUP_value', 'OS_TEM_value', 'OS_INF_value', 'OS_NAS_value', 'OD_SUP_value', 'OD_TEM_value', 'OD_INF_value', 'OD_NAS_value', 'LMac12', 'LMac3', 'LMac6', 'LMac9', 'RMac12', 'RMac3', 'RMac6', 'RMac9', 'OS_macula_csf', 'OS_Volume_Cube', 'OS_Thickness_Avg_Cube', 'OD_macula_csf', 'OD_Volume_Cube', 'OD_Thickness_Avg_Cube', 'GCL_IPL_Avg_OS', 'GCL_IPL_Avg_OD']
    clinical_features = ['Age', 'Sex', 'Race', 'Simp_Race', 'Ethnicity', 'Handedness', 'WTAR_raw', 'WTAR_scaled', 'Corrected_vision', 'OSLogMar', 'ODLogMar', 'BinocularLogMar', 'Education', 'Degree']
    study_features = ['OS_RNFL', 'OS_Cup_Disc_Ratio', 'OS_Cup_Volume', 'OD_RNFL', 'OD_Cup_Disc_Ratio', 'OD_Cup_Volume', 'OS_macula_csf', 'OS_Volume_Cube', 'OD_macula_csf', 'OD_Volume_Cube', 'GCL_IPL_Avg_OS', 'GCL_IPL_Avg_OD']
    study_used_features = ['OS_macula_csf', 'OS_Volume_Cube', 'OD_macula_csf', 'OD_Volume_Cube']

    input_array = StandardScaler().fit_transform(input_array).tolist()
    clf = svm.LinearSVC()
    clf.fit(input_array, labels)
    p = clf.predict(input_array)
    accuracy, precision, recall, f1 = get_metrics(labels, p)
    print("Accuracy, Precision, Recall :", accuracy, precision, recall)
    print("f1:", f1)
    np_input_array = np.array(input_array)
    # print(np_input_array.shape)
    # array_val = input_array.pop(0)
    # np_array_val = np.array(array_val)

    explainer = shap.KernelExplainer(clf._predict_proba_lr, np_input_array)
    shap_values = explainer.shap_values(np_input_array)
    # print(shap_values)
    f = plt.figure()
    # print(shap_values.shape)
    print(shap_values[0].shape)
    x, y = shap_values[0].shape
    for i in range(y):
        print(np.absolute(shap_values[0][:, i]).mean())
    # print(shap_values[0])
    shap.summary_plot(shap_values, np_input_array, plot_type="bar", feature_names=study_used_features)
    # shap.force_plot(explainer.expected_value[0], shap_values[0][0], np_input_array[0], feature_names=oct_features,
    #                 matplotlib=True)


def build_svcs_with_omission(m=1.0, c=1.0):
    is_linear = True
    true_pos = true_neg = false_pos = false_neg = 0
    avg_control_acc = avg_control_prec = avg_control_rec = avg_control_f1 = 0
    avg_pt_acc = avg_pt_prec = avg_pt_rec = avg_pt_f1 = 0

    all_labels = control_labels + pt_labels
    all_input_array = control_data + pt_data
    # all_input_array = StandardScaler().fit_transform(all_input_array).tolist()    # STANDARDIZED SCALING
    for i in range(l_control+l_pt):   # ultimately going up to 68 :0      on 12 for fep d2
        print("Index is:", i)
        removed_label = all_labels[i]
        removed_input = all_input_array[i]
        # print(removed_input)
        # print(len(labels), len(input_array))
        labels = all_labels[:i] + all_labels[i+1:]
        input_array = all_input_array[:i] + all_input_array[i+1:]

        # if is_e2_e3:
        #     if i < l_control:
        #         num_control = l_control - 1
        #         num_pt = l_pt
        #         weights = {0: 0.8, 1: 1.0}
        #     else:
        #         num_control = l_control
        #         num_pt = l_pt - 1
        #         weights = {0: 0.75, 1: 1.0}
        #     # print("m:", m)
        #     # x_weight, y_weight = get_weights(m, num_control, num_pt)
        #     # weights = {0: x_weight, 1: y_weight}
        #
        # print(weights)
        # print("c in build is:", c)
        if is_linear:
            clf = svm.LinearSVC(class_weight='balanced')  # class_weight for data imbalance, gamma='auto' for non-linear
        else:
            clf = svm.SVC(gamma='auto', class_weight='balanced')
        clf.fit(input_array, labels)
        # print(clf.class_weight_)
        p1 = clf.predict([removed_input])
        p = clf.predict(input_array)
        accuracy, precision, recall, f1 = get_metrics(labels, p)
        if i < l_control:
            avg_control_acc += accuracy
            avg_control_prec += precision
            avg_control_rec += recall
            avg_control_f1 += f1
        else:
            avg_pt_acc += accuracy
            avg_pt_prec += precision
            avg_pt_rec += recall
            avg_pt_f1 += f1
        print(p)
        print("Accuracy, Precision, Recall :", accuracy, precision, recall)
        print("f1:", f1)

        truey = removed_label == p1
        print("Removed label:", removed_label, "Predicted label:", p1, "True?", truey)
        if removed_label == 0:
            if truey:
                true_neg += 1
            else:
                false_pos += 1
        else:
            if truey:
                true_pos += 1
            else:
                false_neg += 1
    print_stats(true_neg, true_pos, false_neg, false_pos, l_control, l_pt)
    print("Avg Control Test accuracy, precision, recall, f1:", avg_control_acc/l_control, avg_control_prec/l_control,
          avg_control_rec/l_control, avg_control_f1/l_control)
    print("Avg Patient Test accuracy, precision, recall:", avg_pt_acc/l_pt, avg_pt_prec/l_pt, avg_pt_rec/l_pt,
          avg_pt_f1/l_pt)
    return true_neg, accuracy


def maximize_f1_set(start=95, end=100, incr=10):
    f1_dict = {}
    all_labels = control_labels + pt_labels
    all_input_array = control_data + pt_data
    all_input_array = StandardScaler().fit_transform(all_input_array).tolist()
    for n in range(start, end, incr):
        x_weight = n/100.0
        weights = {0: x_weight, 1: 1.0}
        true_pos = true_neg = false_pos = false_neg = 0
        avg_control_acc = avg_control_prec = avg_control_rec = avg_control_f1 = 0
        avg_pt_acc = avg_pt_prec = avg_pt_rec = avg_pt_f1 = 0
        for i in range(l_control, l_control+l_pt):  # SWITCH
            print("Index is:", i)
            labels = all_labels[:i] + all_labels[i+1:]
            input_array = all_input_array[:i] + all_input_array[i+1:]
            removed_label = all_labels[i]
            removed_input = all_input_array[i]
            print(weights)

            clf = svm.SVC(gamma='auto', class_weight=weights)
            clf.fit(input_array, labels)
            p1 = clf.predict([removed_input])
            p = clf.predict(input_array)
            accuracy, precision, recall, f1 = get_metrics(labels, p)

            if i < l_control:
                avg_control_acc += accuracy
                avg_control_prec += precision
                avg_control_rec += recall
                avg_control_f1 += f1
            else:
                avg_pt_acc += accuracy
                avg_pt_prec += precision
                avg_pt_rec += recall
                avg_pt_f1 += f1
            print(p)
            print("Accuracy, Precision, Recall :", accuracy, precision, recall)
            print("f1:", f1)

            truey = removed_label == p1
            print("Removed label:", removed_label, "Predicted label:", p1, "True?", truey)
            if removed_label == 0:
                if truey:
                    true_neg += 1
                else:
                    false_pos += 1
            else:
                if truey:
                    true_pos += 1
                else:
                    false_neg += 1
        print_stats(true_neg, true_pos, false_neg, false_pos, l_control, l_pt)
        print("Avg Control Test accuracy, precision, recall, f1:", avg_control_acc / l_control,
              avg_control_prec / l_control,
              avg_control_rec / l_control, avg_control_f1 / l_control)
        print("Avg Patient Test accuracy, precision, recall, f1:", avg_pt_acc / l_pt, avg_pt_prec / l_pt,
              avg_pt_rec / l_pt, avg_pt_f1/l_pt)

        f1_dict[x_weight] = [true_pos, avg_pt_f1/l_pt]  # SWITCH BOTH
        # f1_dict[x_weight] = [true_neg, avg_control_f1 / l_control]
    print("F1 DICT...")
    print(f1_dict)

    # if incr != 1:
    #     max_val = max(f1_dict.values())
    #     max_keys = [w for w in f1_dict.keys() if f1_dict[w] == max_val]
    #     if (len(max_keys) == 1):
    #         print(max_keys)
    #         max_key = int(max_keys[0]*100)
    #         if incr == 5:
    #             maximize_f1_set(max_key - 3, max_key + 5, 2)
    #         elif incr == 2:
    #             maximize_f1_set(max_key - 1, max_key + 2, 1)


def print_stats(true_neg, true_pos, false_neg, false_pos, l_control, l_pt):
    print("True neg:", true_neg, "True pos:", true_pos, "False neg:", false_neg, "False pos:", false_pos)
    if true_neg > 0.0 and false_pos > 0.0 and true_pos > 0.0 and false_neg > 0.0:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = (2 * precision * recall) / (precision + recall)
        print("Confusion matrix stats:")
        print("accuracy:", (true_neg + true_pos) / (l_control + l_pt), "precision:", precision)
        print("recall:", recall, "f1:", f1)
    else:
        f1 = None
    return f1


def maximize_f1():
    idx = 2
    removed_label = labels.pop(idx)
    removed_input = input_array.pop(idx)
    m = 0.65
    increment = 0.05
    is_m_peak = False
    while not is_m_peak:
        m_up, m_down = m + increment, m - increment
        f1_dict = {m_down: 0.0, m: 0.0, m_up: 0.0}
        for i in [m_down, m, m_up]:
            x_weight, y_weight = get_weights(i, l_control - 1, l_pt)
            weights = {0: x_weight, 1: y_weight}
            print("weights:", weights)

            clf = svm.SVC(gamma='auto', class_weight=weights)
            clf.fit(input_array, labels)
            p1 = clf.predict([removed_input])
            p = clf.predict(input_array)

            accuracy, precision, recall, f1 = get_metrics(labels, p)
            truey = removed_label == p1
            f1_dict[i] = f1

            print("f1 score:", f1, "accuracy:", accuracy)
            print("Removed label:", removed_label, "Predicted label:", p1, "True?", truey)
        print("F1_DICT:", f1_dict)
        if f1_dict[m_up] <= f1_dict[m] and f1_dict[m_down] <= f1_dict[m]:
            is_m_peak = True
        else:
            if f1_dict[m_up] > f1_dict[m]:
                m = m_up
            elif f1_dict[m_down] > f1_dict[m]:
                m = m_down


def build_pca():
    is_3d = False
    std_input_array = StandardScaler().fit_transform(input_array)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(std_input_array)
    # print(pca.explained_variance_, pca.explained_variance_ratio_)

    fig = plt.figure(figsize=(8, 8))
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('FEP vs Control PCA: All OCT Machine Data', fontsize=20)
    for i in range(len(principalComponents)):
        if i < l_control:
            if is_3d:
                ctrl = ax.scatter(principalComponents[i][0], principalComponents[i][1], principalComponents[i][2], c='r', s=50)
            else:
                ctrl = ax.scatter(principalComponents[i][0], principalComponents[i][1], c='r', s=50)

        else:
            if is_3d:
                pt = ax.scatter(principalComponents[i][0], principalComponents[i][1], principalComponents[i][2], c='g', s=50)
            else:
                pt = ax.scatter(principalComponents[i][0], principalComponents[i][1], c='g', s=50)
    ax.grid()
    ax.legend((ctrl, pt), ('Control', 'FEP'))
    plt.savefig('fep_all_machine_data')
    plt.show()
    plt.clf()
    plt.close()

def used_data_pca():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Average macula_csf')
    ax.set_ylabel('Average volume_cube')
    ax.set_title('FEP vs Control: Volume_Cube and Macula_CSF')
    scatter_vals = []
    for c in control_data:
        csf = (c[0] + c[2]) / 2.0
        vc = (c[1] + c[3]) / 2.0
        scatter_vals.append([csf, vc])
    for p in pt_data:
        csf = (p[0] + p[2]) / 2.0
        vc = (p[1] + p[3]) / 2.0
        scatter_vals.append([csf, vc])
    for i in range(len(scatter_vals)):
        if i < l_control:
            ctrl = ax.scatter(scatter_vals[i][0], scatter_vals[i][1], c='r', s=50)
        else:
            pt = ax.scatter(scatter_vals[i][0], scatter_vals[i][1], c='g', s=50)
    ax.grid()
    ax.legend((ctrl, pt), ('Control', 'FEP'))
    plt.savefig('fep_used_machine_data')
    plt.show()
    plt.clf()
    plt.close()

build_svcs_with_omission()
# build_linear_svc(input_array)
# maximize_f1()
# build_pca()
# maximize_f1_set()
# used_data_pca()

