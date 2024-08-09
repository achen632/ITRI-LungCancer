import pylidc as pl
import cv2
from azure.storage.blob import BlobServiceClient
from pydicom import dcmread
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

# Return annotations based on patient_id and slice_location
def get_anns_from_slice(patient_id, slice_location):
    # Fetch the scan for the patient
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    if not scan:
        raise ValueError(f"No scan found for patient ID {patient_id}")
    # Initialize list to store annotations
    annotations = []
    # Iterate over all annotations in the scan
    for ann in scan.annotations:
        for contour in ann.contours:
            # Check if the annotation is close to the specified slice location
            if abs(contour.image_z_position - slice_location) < scan.slice_spacing:
                annotations.append(ann)
                break
    return annotations

def isNodule(result):
    return len(result.boxes.xywh.tolist()) > 0

# Draws predicted and labelled bboxes on image and saves
def draw_boxes(filename, result, ann):
    # Dont draw if no nodules in either result or ann
    if not isNodule(result) and len(ann) == 0:
        return
    
    image = result.orig_img
    for box in result.boxes.xywh:
        x, y, width, height = box.tolist()
        cv2.circle(image, (int(x+width/2), int(y+height/2)), 15, (0, 255, 0), 2)
    if len(ann) > 0:
        for a in ann:
            y, x, z = a.centroid
            cv2.circle(image, (int(x), int(y)), 25, (0, 0, 255), 2)
    cv2.imwrite(filename, image)

# Returns bounding box from result
def getBBoxes(result):
    boxes = []
    for coords in result.boxes.xywh.tolist():
        x = int(coords[0])
        y = int(coords[1])
        w = int(coords[2])
        h = int(coords[3])
        boxes.append([x, y, w, h])
    return boxes

# Returns if two bounding boxes are the same
def compare_boxes(box1, box2):
    thresh = 15
    return (abs(box1[0]-box2[0]) < thresh and
            abs(box1[1]-box2[1]) < thresh and
            abs(box1[2]-box2[2]) < thresh and
            abs(box1[3]-box2[3]) < thresh)

# Returns if a slice contains a box similar to the reference box
def check_slice_for_box(slice, ref):
    for box in slice:
        if compare_boxes(box, ref):
            return True
    return False

# Checks for one-apart slices
def first_slice_one_apart(index, boxes, ref):
    if index > 0:
        return check_slice_for_box(boxes[index-1], ref)
    return False

# Checks for one-apart slices
def last_slice_one_apart(index, boxes, ref):
    if index < len(boxes)-1:
        return check_slice_for_box(boxes[index+1], ref)
    return False

# Returns index of first slice with a matching box
def get_first_slice(index, boxes, ref):
    while index >= 0:
        if check_slice_for_box(boxes[index], ref):
            index -= 1
        elif first_slice_one_apart(index, boxes, ref):
            index -= 2
        else:
            break
    return index+1

# Returns index of last slice with a matching box
def get_last_slice(index, boxes, ref):
    while index < len(boxes):
        if check_slice_for_box(boxes[index], ref):
            index += 1
        elif last_slice_one_apart(index, boxes, ref):
            index += 2
        else:
            break
    return index-1

# Returns number of slices that contain a box similar to the reference box
def get_num_slices(index, boxes, ref):
    range = get_slice_range(index, boxes, ref)
    return range[1]-range[0]+1

# Returns tuple of first and last slices that contain a box similar to the reference box
def get_slice_range(index, boxes, ref):
    first = get_first_slice(index-1, boxes, ref)
    last = get_last_slice(index+1, boxes, ref)
    return [first, last]

# Returns the max number of contiguous slices that has a matching box
def get_num_relevant_slices(index, boxes):
    max = 0
    for box in boxes[index]:
        num = get_num_slices(index, boxes, box)
        if num > max:
            max = num
    return max

# Extracts bounding box (xywh) info from annotation
def ann_to_bbox(ann):
    if len(ann) == 0:
        return []
    x = []
    y = []
    w = []
    h = []
    for a in ann:
        bbox = a.bbox()
        x.append(bbox[1].start)
        y.append(bbox[0].start)
        w.append(bbox[1].stop - bbox[1].start)
        h.append(bbox[0].stop - bbox[0].start)
    x_avg = int(sum(x)/len(x))
    y_avg = int(sum(y)/len(y))
    w_avg = int(sum(w)/len(w))
    h_avg = int(sum(h)/len(h))
    return [[x_avg, y_avg, w_avg, h_avg]]

# Represents nodule as tuple and adds to set
def add_nodule(set, range, bbox):
    for n in set:
        if compare_boxes(n[1], bbox):
            return
    nod = (tuple(range), tuple(bbox))
    set.add(nod)

# Returns a set of nodules with range of relevant slices and bbox
def get_nodules(indices, preds_filtered):
    nodules = set()
    for i, (index, pred) in enumerate(zip(indices, preds_filtered)):
        for bbox in pred:
            range = get_slice_range(i, preds_filtered, bbox)
            if range[1]-range[0] != 0:
                corrected_range = range[0]-i+index, range[1]-i+index
                add_nodule(nodules, corrected_range, bbox)
    return nodules

# Classifies annotations as benign or malignant (0 or 1) by averaging the radiologists' ratings
def get_class_from_anns(i, anns):
    if anns == []:
        return None
    sum = 0
    for ann in anns:
        sum += ann.malignancy
    return (i+1, sum / len(anns) > 3)

# Adds annotation class to list of classes, recording slice index
def add_cls_from_anns(classifications, i, anns):
    cls = get_class_from_anns(i, anns)
    if cls != None:
        classifications.append(cls)
        
# Calculates the length of overlap between two ranges
def overlap_len(r1, r2):
    start1, end1 = r1
    start2, end2 = r2
    length1 = end1 - start1
    length2 = end2 - start2
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = max(0, overlap_end - overlap_start)
    return overlap_length

# Returns if two nodules are equal
def nods_equal(nod1, nod2):
    box_same = compare_boxes(nod1[1], nod2[1])
    slices_same = overlap_len(nod1[0], nod2[0]) >= 2
    return box_same and slices_same

# Returns the size of the union between two sets
def num_union(p_nods, l_nods):
    count = 0
    for a in p_nods:
        for b in l_nods:
            if nods_equal(a, b):
                count += 1
    return count

# Returns data used for eval metrics
def tp_fp_fn(p_nods, l_nods):
    tp = num_union(p_nods, l_nods)
    fp = len(p_nods) - tp
    fn = len(l_nods) - tp
    return tp, fp, fn

def precision(tp, fp, fn):
    return tp / (tp + fp)

def recall(tp, fp, fn):
    return tp / (tp + fn)

def f1(prec, rec):
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)

# Returns evaluation metrics
def get_metrics(p_nods, l_nods):
    tp, fp, fn = tp_fp_fn(p_nods, l_nods)
    prec = precision(tp, fp, fn)
    rec = recall(tp, fp, fn)
    f1_score = f1(prec, rec)
    return prec, rec, f1_score

# Evaluate results
def evaluate(results):
    tp = 0
    fp = 0
    fn = 0
    for res in results:
        p_nods, l_nods = res
        metrics = help.tp_fp_fn(p_nods, l_nods)
        tp += metrics[0]
        fp += metrics[1]
        fn += metrics[2]
    prec = help.precision(tp, fp, fn)
    rec = help.recall(tp, fp, fn)
    f1_score = help.f1(prec, rec)
    return prec, rec, f1_score

# Retrieves DICOM file from Azure Blob Storage
def get_dicom(container_client, blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    blob_data = blob_client.download_blob().readall()
    blob_stream = BytesIO(blob_data)
    return dcmread(blob_stream)

# Return azure blob container client
def get_container_client():
    with open('/home/andrew/ITRI-LungCancer/keys.txt', 'r') as file:
        data = file.read().splitlines()
        account_name    = data[0]
        account_key     = data[1]
        container_name  = data[2]
    blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)
    return blob_service_client.get_container_client(container_name)

# Returns list of blob names
def get_blob_list(blob_container_client, patient_id):
    return blob_container_client.list_blob_names(name_starts_with=patient_id)

# Apply windowing and normalization to img
def window_img(img, window_center, window_width):
    win_min = window_center - window_width / 2.0
    win_max = window_center + window_width / 2.0
    img = np.clip(img, win_min, win_max)
    img = (img - win_min) / (win_max - win_min)
    img = np.uint8(img * 255)
    return img

# Apply rescaling to img based on dicom metadata
def rescale_img(ds, img):
    if 'RescaleIntercept' in ds and 'RescaleSlope' in ds:
        img = img * ds.RescaleSlope + ds.RescaleIntercept
    return img

# Returns np array after processing image from dicom
def dicom_to_img(dicom):
    image_base = rescale_img(dicom, dicom.pixel_array)
    return window_img(image_base, -300, 2000)

# Changes dcm file number by val
def change_file_num(blob_name, val):
    path = blob_name[0:-7]
    num = int(blob_name[-7:-4])
    return path+str(num+val).zfill(3)+'.dcm'
    
# Returns image from blob
def get_image_from_blob(blob_container_client, blob_name):
    try:
        dicom = get_dicom(blob_container_client, blob_name)
    except:
        return np.zeros((512, 512), dtype=np.uint8)
    return dicom_to_img(dicom)

# Returns context images +/- 2 slices away from the center slice
def get_context_imgs(blob_container_client, blob_name):
    image_prev = get_image_from_blob(blob_container_client, change_file_num(blob_name, -2))
    image_next = get_image_from_blob(blob_container_client, change_file_num(blob_name, 2))
    return image_prev, image_next

# Plots nodules on histogram and saves to out_path
def save_histogram(out_path, indices, preds_filtered, labels_filtered, patient_id):
    # Check for empty lists
    if indices == []:
        return
    
    # Determines max y value to show on graph
    y_min = indices[0]
    y_max = indices[-1]

    # Create an array of zeros for the range 0-200
    pred_presence = [0] * (y_max-y_min)
    label_presence = [0] * (y_max-y_min)

    # Mark presence of each value in the list
    for i, pred, label in zip(indices, preds_filtered, labels_filtered):
        if pred != []:
            pred_presence[i-y_max] = 1
        if label != []:
            label_presence[i-y_max] = 1

    # Plot
    plt.figure(figsize=(6, 3))
    plt.bar(range(y_min, y_max), label_presence, width=1.0, alpha=.5, label='Label')
    plt.bar(range(y_min, y_max), pred_presence, width=1.0, alpha=.5, label='Prediction')
    plt.xlabel('Slice Num')
    plt.ylabel('Nodule Presence (1 if present, 0 if not)')
    plt.title(f'Labeled and Predicted Nodules from slices {y_min} to {y_max} for {patient_id}')
    plt.xticks(range(y_min, y_max, 10))
    plt.legend()
    plt.savefig(f'{out_path}/{patient_id}.png')