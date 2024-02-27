from dataproc import *
import os
import torch
import numpy as np
import random

def parse_annotations(file2mapping, single_st_ann, annotations_map, ss_vid_ann, single_student_tids):
    #crowd
    exercise2points = dict()
    exercise2errors = dict()
    encoded_errors = {"нет ошибок":0}
    for filename in file2mapping:
        sid2kps = parse_crowd_exercise(filename, file2mapping[filename])    
        with open(annotations_map[filename], "r", encoding="utf8") as ja:
            ann = json.load(ja)
            for k in ann:
                people = list(ann[k].keys())
                for personid in people:
                    intkey = int(personid)
                    ann[k][intkey] = ann[k][personid]            
        exname = os.path.basename(filename).split('.')[0]
        student_tries = split_student_tries(sid2kps, ann['people_exercise_frames'])
        errs = get_error_codes(ann, encoded_errors)
        exercise2points[exname] = student_tries
        exercise2errors[exname] = errs
    #single student
    student2points = dict()
    #student2errors = dict()
    with open(single_st_ann, "r", encoding="utf8") as common_ann_file:
        common_ann = json.load(common_ann_file)
    # ss_vid_ann = [os.path.join("../alphapose_res", f) for f in ["0756_alpha.json", "0912_alpha.json",
    #                                                             "1043_alpha.json", "1156_alpha.json", 
    #                                                             "1237_alpha.json"]]
    for filename in ss_vid_ann:
        stud_id = os.path.split(filename)[-1][:4]
        dynamic = parse_single_person_exercise(filename, single_student_tids[stud_id])
        internal_vid_id = common_ann["file2id"][stud_id]
        student_tries = split_student_tries({internal_vid_id:dynamic}, common_ann['people_exercise_frames'])
        student2points[internal_vid_id] = student_tries[internal_vid_id]
    student2errors = get_error_codes(common_ann, encoded_errors)
    #cleaning
    delete_keys = []
    for ename in exercise2errors:
        for k in exercise2errors[ename]:
            if type(k) == str:
                delete_keys.append(k)
        for k in delete_keys:
            del exercise2errors[ename][k]
        delete_keys.clear()  
    # errors
    idx2error = dict()
    for n in encoded_errors:
        idx = encoded_errors[n]
        idx2error[idx] = n
    exercise2points["single_student"] = student2points
    exercise2errors["single_student"] = student2errors
    return idx2error, exercise2points, exercise2errors

def encode_label_tensor(label:list, class_count:int):
    target = torch.zeros(class_count)
    for idx in label:
        target[idx] = 1
    return target

def encode_label_array(label:list, class_count:int):
    target = np.zeros(class_count)
    for idx in label:
        target[idx] = 1
    return target
    
def decode_target(target, classes:dict, threshold:float=0.5):
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
          result.append(classes[i])
    return result

def random_mirroring(img_array:np.ndarray, channel:int, proba:float=0.5)->np.ndarray:
    if torch.rand(1) <= proba: # apply with probability
        return mirror_coordinate(img_array, channel)
    else: # else - return unchanged
        return img_array

def normalize01(img_array:np.ndarray)->np.ndarray:
    res = np.copy(img_array)
    for channel in range(res.shape[-1]):
        charr = res[:, :, channel]
        res[:, :, channel] = (charr - charr.min()) / (charr.max() - charr.min())
    return res

# splits
def fraction_split(exercise2points:dict, exercise2errors:dict, train_frac:float, val_frac:float, test_frac:float, state=0):
    random.seed(state)
    assert (train_frac + val_frac + test_frac) == 1
    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []
    val_imgs, val_labels = [], []
    
    train_ids = []
    val_ids = []
    test_ids = []
    total_data_count = 0
    for exname in exercise2points:
        cur_exerc = exercise2points[exname]
        cur_errors = exercise2errors[exname]
        for p in cur_exerc:
            tries = cur_exerc[p]
            errs = cur_errors[p]
            for i, e in enumerate(tries):
                total_data_count += 1
                rval = random.random()
                if rval <= train_frac:
                    train_imgs.append(e)
                    train_labels.append(errs[i])
                    train_ids.append((exname, p, i))
                elif rval <= (train_frac + test_frac):
                    test_imgs.append(e)
                    test_labels.append(errs[i])
                    test_ids.append((exname, p, i))
                else:
                    val_imgs.append(e)
                    val_labels.append(errs[i])
                    val_ids.append((exname, p, i))
    return [(train_imgs, train_labels), (test_imgs, test_labels), (val_imgs, val_labels)]

def pick_elems_on_indices_from_both(data, labels, indices):
    X = []
    y = []
    for i in indices:
        X.append(data[i])
        y.append(labels[i])
    return X, y

def stratify_split(exercise2points:dict, exercise2errors:dict, train_frac:float, val_frac:float, test_frac:float, class_count:int, state=0):    
    random.seed(state)
    input_data = []
    labels = []
    total_count = 0
    class2idx = dict()
    for exname in exercise2points:
        cur_exerc = exercise2points[exname]
        cur_errors = exercise2errors[exname]
        for p in cur_exerc:
            tries = cur_exerc[p]
            errs = cur_errors[p]
            for i, e in enumerate(tries):
                input_data.append(e)
                labels.append(errs[i])
                #labels.append(encode_label(errs[i], class_count).numpy())
                for error in errs[i]:
                    if error not in class2idx:
                        class2idx[error] = []
                    class2idx[error].append(total_count)
                total_count += 1    
    train_instances_idx = set()
    val_instances_idx = set()
    test_instances_idx = set()
    for cl in class2idx:
        class_indices = class2idx[cl]
        random.shuffle(class_indices)
        last_train_idx = max(int(train_frac * len(class_indices)), 1)
        train_picks = class_indices[:last_train_idx]
        train_instances_idx.update(train_picks)
        if (len(class_indices) - len(train_picks)) == 0:
            continue
        last_test_idx = last_train_idx + max(int(test_frac * len(class_indices)), 1)
        test_picks = class_indices[last_train_idx:last_test_idx]
        test_instances_idx.update(test_picks)
        if (len(class_indices) - len(train_picks) - len(test_picks)) == 0:
            continue
        val_instances_idx.update(class_indices[last_test_idx:])
    # clear repeating values
    val_instances_idx = val_instances_idx - val_instances_idx.intersection(train_instances_idx)
    test_instances_idx = test_instances_idx - test_instances_idx.intersection(train_instances_idx)
    
    X_train, y_train = pick_elems_on_indices_from_both(input_data, labels, train_instances_idx)  
    X_test, y_test = pick_elems_on_indices_from_both(input_data, labels, test_instances_idx)
    X_val, y_val = pick_elems_on_indices_from_both(input_data, labels, val_instances_idx)
    
    # train_count = round(total_count * train_frac)
    # val_count = round(total_count * val_frac)
    # test_count = round(total_count * test_frac)
    # "The minimum number of groups for any class cannot be less than 2" для sklearn.model_selection.train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(input_data, 
    #                                                     train_size=train_count+val_count, 
    #                                                     random_state=state, stratify=labels)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, 
    #                                                   train_size=train_count, 
    #                                                   random_state=state, stratify=y_train)
    return [(X_train, y_train), (X_test, y_test), (X_val, y_val)]