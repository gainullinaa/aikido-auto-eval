import numpy as np
import cv2 as cv

import json
import os

def bbox_iou(box1, box2) -> float: 
    """ intersection-over-union [0...1]"""
    #BoundingBox(origin_x, origin_y, width, height)
    xmax1 = box1[0] + box1[2]
    ymax1 = box1[1] + box1[3]
    xmax2 = box2[0] + box2[2]
    ymax2 = box2[1] + box2[3]
    dx = min(xmax1, xmax2) - max(box1[0], box2[0])
    dy = min(ymax1, ymax2) - max(box1[1], box2[1])
    if (dx>=0) and (dy>=0):
        inters = dx*dy
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        percent_coverage = inters / (area1 + area2 - inters)
        return percent_coverage
    return 0
    
def update_person_box(prev:dict, newly:list)-> dict: 
    """ Назначает на id из прошлого кадра новые ограничивающие рамки в текущем кадре. 
        Не меняет prev.
        Args:
            prev, dict - id человека:рамка из прошлого кадра
            newly, list - список рамок на текущем кадре
        Returns:
            dict, returned_dict.keys() == prev.keys()"""
    new_index = dict()
    ious = dict()
    for box in newly:
        potential_index = dict()
        for idx in prev:
            cur_iou = bbox_iou(box, prev[idx])
            if cur_iou <= 1e-4: # almost 0
                continue
            if idx in new_index and (ious[idx] >= cur_iou):
                continue                
            potential_index[idx] = cur_iou
        if len(potential_index) > 0:
            max_index = max(potential_index, key=potential_index.get)
            new_index[max_index] = box
            ious[max_index] = potential_index[max_index]
        else: # found new person
            new_max_idx = max(prev) + 1
            new_index[new_max_idx] = box
            ious[new_max_idx] = 1
    for idx in prev:
        if not (idx in new_index):
            new_index[idx] = prev[idx]
    return new_index

def get_tracked_boxes_on_frames(keypoints_info:list) -> dict:
    """ Возвращает все рамки по кадрам в форме словаря.
        Args:
            keypoints_info, list - из аннотаций после alphapose детектора ключевых точек позы
        Returns:
            dict, frame_index:bboxes_list
    """
    total_frames = keypoints_info[-1]["image_id"] + 1 
    current_json_entry = 0
    boxes_by_frame = dict()
    for frame_idx in range(total_frames):
        frame_people = []
        while keypoints_info[current_json_entry]['image_id'] == frame_idx:
            box = keypoints_info[current_json_entry]['box']
            frame_people.append(box)
            if current_json_entry + 1 >= len(keypoints_info): break
            current_json_entry += 1
        boxes_by_frame[frame_idx] = frame_people
        
    sorting_key = lambda b: b[2] * b[3]
    frame0boxes_by_area = sorted(boxes_by_frame[0], key=sorting_key, reverse=True)
    prev_boxs = dict()
    for i, v in enumerate(frame0boxes_by_area):#[:students_num]):
        prev_boxs[i] = v
    frame2boxes = {0 : prev_boxs}
    for i in range(1, total_frames):
        frame_boxes_by_area = sorted(boxes_by_frame[i], key=sorting_key, reverse=True)
        prev_boxs = update_person_box(prev_boxs, frame_boxes_by_area)
        frame2boxes[i] = prev_boxs
    return frame2boxes

def get_keypoints_by_student(studentids:set, kps_info:list, joint_count:int=26)-> dict: 
    """ Возвращает координаты ключевых точек по кадрам в словаре по ученику
        Args:
            studentids, set - id учеников
            kps_info, list - аннотации после alphapose с id учеников
            joint_count, int - чило ключевых точек
        Returns:
            dict, student_id:ndarray, ndarray.shape==(joint_count, frame_len, 3)
    """
    max_frame_count = kps_info[-1]["image_id"] + 1
    studentid2dynamic = dict() # student to frame entry
    for k in studentids:
        studentid2dynamic[k] = []
    for entry in kps_info:
        try:
            if type(entry["idx"]) == list:
                idx = entry["idx"][0]
            if type(entry["idx"]) == int:
                idx = entry["idx"]
            else:
                raise TypeError
            if idx in studentid2dynamic:
                studentid2dynamic[idx].append(entry)
        except TypeError:
            pass
    
    sid2kps = dict()
    for student_id in studentid2dynamic:
        dynamic_info = studentid2dynamic[student_id]
        cur_student_kps = [[] for i in range(joint_count)] # joints
        prev_frame = 0
        for pose in dynamic_info:
            if (pose['image_id'] - prev_frame) > 1:
                stop_frame = pose['image_id'] - 1
                while prev_frame < stop_frame:
                    for jlist in cur_student_kps:
                        jlist.append([0, 0, 0])
                    prev_frame += 1
            pps = pose["keypoints"]
            for jidx, jlist in enumerate(cur_student_kps):
                # scale coordinates to box size
                x = (pps[jidx*3] - pose['box'][0]) / float(pose['box'][2]) 
                y = (pps[1 + jidx*3] - pose['box'][1]) / float(pose['box'][3])
                conf = pps[2 + jidx*3]
                jlist.append([x, y, conf])
            prev_frame = pose['image_id']
        joints_array = np.array(cur_student_kps)
        if joints_array.shape[1] < max_frame_count:
            add_fr = max_frame_count - joints_array.shape[1]
            addition = np.zeros((joints_array.shape[0], add_fr, joints_array.shape[2]))
            joints_array = np.concatenate((joints_array, addition), axis=1)
        elif joints_array.shape[1] > max_frame_count:
            joints_array = joints_array[:, :max_frame_count, :]
        sid2kps[student_id] = joints_array
    return sid2kps
        
def parse_crowd_exercise(json_ann_path:str, track2student:dict)-> dict: 
    """ Парсит json-аннотацию в словарь координат-массивов numpy.
        Args:
            json_ann_path, str - путь до аннотации
            track2student, dict - словарь автоопределенных id:id ученика
        Returns:
            dict, student_id:ndarray, ndarray.shape==(joint_count, frame_len, 3)
    """
    with open(json_ann_path, "r") as res_file:
        kps_info = json.load(res_file)
    frame2boxes = get_tracked_boxes_on_frames(kps_info)
    # autotracked id to student id
    json_entry_count = 0
    for fid in frame2boxes:
        boxes = frame2boxes[fid]
        # get entries
        frame_entries = []
        while json_entry_count < len(kps_info) and kps_info[json_entry_count]["image_id"] == fid:
            frame_entries.append(kps_info[json_entry_count])
            json_entry_count += 1    
        for entry in frame_entries: 
            for det_id in boxes:
                if not det_id in track2student: 
                    student_id = -1
                else:
                    student_id = track2student[det_id]
                box = boxes[det_id]            
                entbox = entry["box"]
                if entbox[0] != box[0]: continue # bad coding
                if entbox[1] != box[1]: continue
                if entbox[2] != box[2]: continue
                if entbox[3] != box[3]: continue
                entry["idx"] = student_id
    # get keypoints as array in dictionary by student id
    return get_keypoints_by_student(set(track2student.values()), kps_info)

def parse_single_person_exercise(json_ann_path:str, pick_tracked_id:int=0)->np.ndarray:
    """ Парсит аннотацию видео одного человека в массив ndarray.
        Args:
            json_ann_path, str - путь до аннотации
            pick_tracked_id, int, optional - человека по какому id после отслеживания парсить, default=0
        Returns:
            numpy.ndarray ключевых точек из аннотации
    """
    with open(json_ann_path, "r") as res_file:
        kps_info = json.load(res_file)
    frame2boxes = get_tracked_boxes_on_frames(kps_info)
    json_entry_count = 0
    person_entries = []
    for fid in frame2boxes:
        boxes = frame2boxes[fid]
        frame_entries = []
        while json_entry_count < len(kps_info) and kps_info[json_entry_count]["image_id"] == fid:
            frame_entries.append(kps_info[json_entry_count])
            json_entry_count += 1 
        for entry in frame_entries: 
            for det_id in boxes:
                if det_id == pick_tracked_id:
                    box = boxes[det_id]            
                    entbox = entry["box"]
                    if entbox[0] != box[0]: break # bad coding
                    if entbox[1] != box[1]: break
                    if entbox[2] != box[2]: break
                    if entbox[3] != box[3]: break
                    entry["idx"] = pick_tracked_id
                    person_entries.append(entry)
                else:
                    continue
    one_student_dict = get_keypoints_by_student({pick_tracked_id}, person_entries)
    # return 1 array of points of 1 person
    return one_student_dict[pick_tracked_id]        

def get_error_codes(annotations:dict, mistakes_dict:dict, correct_code:int=0)->dict:
    """ Возвращает словарь student_id:mistakes. 
        Значение - список списков с кодом ошибки для каждой попытки упражнения.
        Args:
            annotations, dict - ручные аннотации
            mistakes_dict, dict - mistake_name_string:mistake_code_int
            correct_code, int - no-mistakes code
        Returns:
            dict, student_id:mistakes_list
    """
    all_people = annotations["people_exercise_frames"].keys()
    errors_map = dict()
    for p in all_people:
        if p not in annotations["exercise_errors"]: # no mistakes in this person exercises
            ex_count = len(annotations["people_exercise_frames"][p])
            errors_map[p] = [[correct_code] for _ in range(ex_count)]
        else:
            ex_count = len(annotations["people_exercise_frames"][p])
            tries_with_mistakes = dict()
            for mistake in annotations["exercise_errors"][p]:
                if mistake["er_type"] not in mistakes_dict:
                    if len(mistakes_dict) == 0:
                        mistakes_dict[mistake["er_type"]] = correct_code + 1
                    else:
                        last_code = max(mistakes_dict.values())
                        mistakes_dict[mistake["er_type"]] = last_code + 1
                encoded = mistakes_dict[mistake["er_type"]]
                for t in mistake["tries_index"]:
                    if t in tries_with_mistakes:
                        tries_with_mistakes[t].append(encoded)
                    else:
                        tries_with_mistakes[t] = [encoded]
            errors = []
            for tcount in range(ex_count):
                if tcount in tries_with_mistakes:
                    errors.append(tries_with_mistakes[tcount])
                else:
                    errors.append([correct_code])
            errors_map[p] = errors
    return errors_map

def split_student_tries(student_points:dict, exercise_ends:dict)->dict:
    """ Разделяет всю длительность ключевых точек на разные попытки.
        Итоговый словарь содержит список view-массивов в значении.
        !Обязательно student_points.keys()==exercise_ends.keys() 
        Args:
           student_points, dict - student_id:keypoints_nparray 
           exercise_ends, dict - student_id:[[start, end], [start, end], ...]
        Returns:
            dict, student_id:[nparray1, nparray2, ...]
    """
    student_split_ex = dict()
    for student in student_points:
        cur_kps = student_points[student]
        cur_ex_ends = exercise_ends[student]        
        tries_chunks = []
        for start, end in cur_ex_ends: # exercise endframe is inclusive
            tries_chunks.append(cur_kps[:, start:end+1, :])
        student_split_ex[student] = tries_chunks
    return student_split_ex

def resample_to_frame_length(keypoints:np.array, new_length:int)->np.array:
    """ Assumes keypoints.shape = (n joints, n frames, n coords).
        Returned shape = (n joints, new_length, n coords).
    """
    if keypoints.shape[1] == new_length: # nothing to change
        return keypoints
    if keypoints.shape[1] < new_length: # append 0
        add_frames = new_length - keypoints.shape[1]
        addition = np.zeros((keypoints.shape[0], add_frames, keypoints.shape[2]),
                           dtype=keypoints.dtype)
        return np.concatenate((keypoints, addition), 1)        
    # n frames > new_length
    picked_frame_idx = np.linspace(0, keypoints.shape[1] + 1, new_length, 
                                   endpoint=False, dtype=np.int64)    
    return np.copy(keypoints[:, picked_frame_idx, :])

def split_into_equal_frame_len(student_points:list, borders:list, piece_len:int, not_exercise_val:int=0, drop_last:bool=True):
    """ Разделяет цельную длительность движений каждого ученика на куски одной длины.
        Разбиение происходит по 2ой размерности (shape[1])
        Args:
            student_points, list[np.ndarray] - список массивов ключевых точек произвольной длительности (shape=(26, frame_len, 3 or 2)) 
            borders, list[np.ndarray] - список 1d-массивов размеченных границ упражнений, может быть None
                len(student_points) == len(borders)
                student_points[n].shape[1] == len(borders[n]) (0 <= n < len(student_points))
            piece_len, int - какой длины должны быть куски
            not_exercise_val, int - тег для кадров, в которых не происходит упражнение
            drop_last, bool - оставлять ли "хвосты", которые меньше длины piece_len, дополняя их паддингом            
        Returns:
            split_student_points, dict[int, list[np.ndarray]] - ключом является индекс в student_points (номер ученика), 
                значением - список массивов точек одинаковой длины в кадрах
            if borders != None: split_borders, dict[int, list[np.ndarray]] - keys()==split_student_points.keys(),
                значение - список массивов размеченных границ
    """
    student2pieces = dict()
    student2border_pieces = dict()
    for sidx, points in enumerate(student_points):
        if borders: 
            cur_borders = borders[sidx]
        dynamic_pieces = []
        border_pieces = []
        for start_idx in range(0, points.shape[1], piece_len):
            if (start_idx + piece_len) >= points.shape[1]: # tail
                if drop_last: break
                else:
                    padding_size = piece_len - (points.shape[1] - start_idx)
                    # points
                    pad = np.zeros((points.shape[0], padding_size, points.shape[-1]))
                    dynamic_pieces.append(np.concatenate((points[:, start_idx:, :], pad), axis=1))
                    # borders
                    if borders: 
                        border_pad = np.full(padding_size, not_exercise_val)
                        border_pieces.append(np.concatenate((cur_borders[start_idx:], border_pad)))
            else:
                # points
                points_piece = np.copy(points[:, start_idx:start_idx + piece_len, :])
                dynamic_pieces.append(points_piece)
                # borders
                if borders: 
                    border_piece = np.copy(cur_borders[start_idx:start_idx + piece_len])
                    border_pieces.append(border_piece)
        student2pieces[sidx] = dynamic_pieces
        if borders: 
            student2border_pieces[sidx] = border_pieces
    if borders: 
        return student2pieces, student2border_pieces
    return student2pieces


