from dataproc import *
import os
import random
import copy

import torch
import numpy as np

def parse_annotations(file2mapping:dict, single_st_ann:str, annotations_map:dict, ss_vid_ann:list, single_student_tids:dict, file2exercise:dict):
    """ Парсит все ручные аннотации с ошибками и окончанием вместе с автоматически полученными аннотациями 
        координат ключевых точек, собирая всё в словари с раздельными попытками упражнений в виде массивов.
        Args:
            file2mapping, dict[str, dict[int, int]] - путь до файла с детектированными координатами к. точек (ключ) 
                : словарь автоопределенных id при трекинге человека вручную проставленным id в аннотациях ошибок и границ упражнений (значение)
            single_st_ann, str - путь до файла с аннотациями ошибок и границ упражнений видео с одним учеником
            annotations_map, dict[str, str] - путь до json-файлов с координатами точек к пути до ручных аннотаций ошибок и границ упражнений
            ss_vid_ann, list - список путей до json-файлов с координатами ключевых точек, полученных от видео с одним учеником
            single_student_tids, dict[str, int] - номер из видео с одним человеком (ключ) к id от автотрекинга человека(значение)
            file2exercise, dict[str, str] - путь до файла с детектированными координатами к. точек(ключ) к названию упражнения(значение)

        file2mapping.keys()==annotations_map.keys()==file2exercise.keys()

        Returns:
            (idx2error, exercise2points, exercise2errors)
            idx2error, dict[int, str] - маппинг код ошибки к её названию
            exercise2points, dict[str, dict[int, list[np.ndarrays]]] - имя упражнения 
                к словарю с id ученика в качестве ключа со списком его попыток этого упражнения в качестве значения
            exercise2errors, dict[str, dict[int, list[list[int]]]] - имя упражнения к словарю (id ученика:список ошибок к каждой его попытке)
            Длина списков-значений в exercise2points и exercise2errors одинакова.
    """
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
        exname = file2exercise[filename] #os.path.basename(filename).split('.')[0]
        student_tries = split_student_tries(sid2kps, ann['people_exercise_frames'])
        errs = get_error_codes(ann, encoded_errors)
        exercise2points[exname] = student_tries
        exercise2errors[exname] = errs
    #single student
    with open(single_st_ann, "r", encoding="utf8") as common_ann_file:
        common_ann = json.load(common_ann_file)
    student2errors = get_error_codes(common_ann, encoded_errors)
    for filename in ss_vid_ann:
        stud_id = os.path.split(filename)[-1][:4]
        dynamic = parse_single_person_exercise(filename, single_student_tids[stud_id])
        internal_vid_id = common_ann["file2id"][stud_id]
        student_tries = split_student_tries({internal_vid_id:dynamic}, common_ann['people_exercise_frames'])
        stud_exercs_names = common_ann["exercise_names"][internal_vid_id]
        stud_errs = student2errors[internal_vid_id]
        exercise_ids = {en:(max(exercise2points[en])+1) for en in exercise2points}
        for i, ename in enumerate(stud_exercs_names):        
            if ename in exercise2points:
                new_stud_code = exercise_ids[ename]
                if new_stud_code not in exercise2points[ename]:
                    exercise2points[ename][new_stud_code] = []
                    exercise2errors[ename][new_stud_code] = []
                exercise2points[ename][new_stud_code].append(student_tries[internal_vid_id][i])
                # errors update
                exercise2errors[ename][new_stud_code].append(stud_errs[i])
            else:
                exercise_ids[ename] = int(internal_vid_id)
                exercise2points[ename] = dict()
                exercise2points[ename][exercise_ids[ename]] = []
                exercise2points[ename][exercise_ids[ename]].append(student_tries[internal_vid_id][i])            
                # errors update
                exercise2errors[ename] = dict()
                exercise2errors[ename][exercise_ids[ename]] = []
                exercise2errors[ename][exercise_ids[ename]].append(stud_errs[i])
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

# augmentations
def mirror_coordinate(coords:np.ndarray, coord_idx:int=0):
    """ Принимает, что координата должна быть в основном в 0...1, но может быть >1 и <0.
        Отражает относительно 0.5, меняется только в 3 размерности (coords.shape[2]), как цветовой канал. 
        Не для нормализованных данных!
        Args:
            coords, ndarray - len(coords.shape)==3
            coord_idx, int - какую координату нужно отразить
        Returns:
            np.copy(coords), но соответствующая координата отражена
    """
    new_coords = np.copy(coords)
    new_coords[:, :, coord_idx] = 1 - new_coords[:, :, coord_idx]
    return new_coords   

def batch_second_collate(batch):
    """ Замена collate_fn для dataloader'ов, если обучается LSTM-модель с batch_first=False.
    """
    data = []
    labels = []
    for i, t in batch:
        data.append(torch.tensor(i))
        labels.append(torch.tensor(t))
    stacked_data = torch.cat(data, 1)
    stacked_labels = torch.stack(labels)
    return stacked_data, stacked_labels

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

# splitting exercise tries

def parse_borders_annotations(file2mapping:dict, single_st_ann:str, annotations_map:dict, ss_vid_ann:list, single_student_tids:dict):
    """ Парсит ручные и автоматические аннотации в список ключевых точек с полной длительности видео и границ (начала и конца).
        Args:
            file2mapping, dict[str, dict[int, int]] - путь до файла с детектированными координатами к. точек (ключ) 
                : словарь автоопределенных id при трекинге человека вручную проставленным id в аннотациях ошибок и границ упражнений (значение)
            single_st_ann, str - путь до файла с аннотациями ошибок и границ упражнений видео с одним учеником
            annotations_map, dict[str, str] - путь до json-файлов с координатами точек к пути до ручных аннотаций ошибок и границ упражнений
            ss_vid_ann, list - список путей до json-файлов с координатами ключевых точек, полученных от видео с одним учеником
            single_student_tids, dict[str, int] - номер из видео с одним человеком (ключ) к id от автотрекинга человека(значение)

        file2mapping.keys()==annotations_map.keys()
        Returns:
            (student_exercise_points, exercise_borders)
            student_exercise_points, list[np.ndarray] - список полных координат каждого студента,
                форма каждого элемента: (26, длительность видео в кадрах, 3)
            exercise_borders, list[list[list[int]]] - список списков индексов начальных и конечных кадров упражнения по студенту,
                каждый элемент списка первого уровня имеет длину 2
    """
    #crowd
    student_exercise_points = []
    exercise_borders = []
    for filename in file2mapping:
        sid2kps = parse_crowd_exercise(filename, file2mapping[filename])    
        with open(annotations_map[filename], "r", encoding="utf8") as ja:
            ann = json.load(ja)
        for sid in sid2kps:
            exerc_arr = sid2kps[sid]
            student_exercise_points.append(exerc_arr)
            exercise_borders.append(ann["people_exercise_frames"][str(sid)])
    #single student
    with open(single_st_ann, "r", encoding="utf8") as common_ann_file:
        common_ann = json.load(common_ann_file)
    for filename in ss_vid_ann:
        stud_id = os.path.split(filename)[-1][:4]
        dynamic = parse_single_person_exercise(filename, single_student_tids[stud_id])
        internal_vid_id = common_ann["file2id"][stud_id]
        student_exercise_points.append(dynamic)
        exercise_borders.append(common_ann["people_exercise_frames"][internal_vid_id])
    return student_exercise_points, exercise_borders

EXTOKENS = {"<NOT_EXERCISE>":0, "<START>":1, "<EXERCISE_BODY>":2, "<END>":3, "<END_START>":4, "<EDGE>":5}

def encode_borders(students_borders:list, frame_durations:list, fill_exercise_token:str, equal_edges:bool=False, diff_borders_one_frame:str="move_start"):
    """ Кодирует границы в вектор тегов длиной frame_durations.
        Args:
            students_borders, list[list[list[int]]] - список по ученику начальных и конечных кадров упражнений в общей записи
            frame_durations, list[int] - список длительности записи по ученику
            fill_exercise_token, str - чем заполнять промежуток между началом и концом упражнения; 
                допускаются None, '<NOT_EXERCISE>', '<EXERCISE_BODY>'
            equal_edges, bool - кодировать ли начало и конец одним значением, 
                if True - ignore diff_borders_one_frame
            diff_borders_one_frame, str - стратегия, если на одном кадре есть конец упражнения и начало следующего:
                'move_start' - сдвигать только начало следующего упражнения на следующий кадр
                'move_end' - сдвигать только конец текущего упражнения на предыдущий кадр
                'token' - использовать специальный токен
        Returns:
            encoded_borders, list[np.ndarray] - список по ученику границ в виде вектора
                len(encoded_borders)==len(students_borders)
    """
    if not fill_exercise_token:
        fill_exercise_token = '<NOT_EXERCISE>'
    assert fill_exercise_token=='<NOT_EXERCISE>' or fill_exercise_token=='<EXERCISE_BODY>'
    assert diff_borders_one_frame=='move_start' or diff_borders_one_frame=='move_end' or diff_borders_one_frame=='token'
    students_borders = copy.deepcopy(students_borders)
    encoded_borders = []
    for student, st_borders in enumerate(students_borders):
        edges_container = np.full(frame_durations[student], EXTOKENS["<NOT_EXERCISE>"])
        was_overlapping = False
        for idx, borders in enumerate(st_borders):
            end_frame_idx = min(borders[1], len(edges_container)-1)
            if equal_edges:
                edges_container[borders[0]] = EXTOKENS["<EDGE>"]
                edges_container[end_frame_idx] = EXTOKENS["<EDGE>"]            
            else:
                # this exercise end is on same frame as the next exercise frame
                is_overlapping = (idx + 1 < len(st_borders)) and (st_borders[idx + 1][0] == borders[1])
                if diff_borders_one_frame == 'token':
                    if not was_overlapping:
                        edges_container[borders[0]] = EXTOKENS["<START>"]
                    if is_overlapping:
                        edges_container[end_frame_idx] = EXTOKENS["<END_START>"]
                    else:
                        edges_container[end_frame_idx] = EXTOKENS["<END>"]
                else:
                    if is_overlapping and (diff_borders_one_frame == 'move_start'):
                        st_borders[idx + 1][0] += 1
                    elif is_overlapping and (diff_borders_one_frame == 'move_end'):
                        borders[1] -= 1
                        end_frame_idx = min(borders[1], len(edges_container)-1)
                    edges_container[borders[0]] = EXTOKENS["<START>"]
                    edges_container[end_frame_idx] = EXTOKENS["<END>"]
                was_overlapping = is_overlapping
                # put start edge
                # if diff_borders_one_frame != 'token':
                #     edges_container[borders[0]] = EXTOKENS["<START>"]
                #put end edge
                # if idx + 1 < len(st_borders): # not last exercise, so check borders overlap
                #     next_ex_borders = st_borders[idx + 1]
                #     if next_ex_borders[0] == borders[1]: #frame overlap                
                #         if diff_borders_one_frame == 'move_start':
                #             next_ex_borders[0] += 1
                #         elif diff_borders_one_frame == 'move_end':
                #             borders[1] -= 1            
                # edges_container[end_frame_idx] = EXTOKENS["<END>"]
                # if diff_borders_one_frame == 'token':
                #     edges_container[end_frame_idx] = EXTOKENS["<END_START>"]
            # fill exercise frames    
            edges_container[borders[0] + 1 : end_frame_idx] = EXTOKENS[fill_exercise_token]
        encoded_borders.append(edges_container)
    return encoded_borders






