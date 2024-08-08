import pylidc as pl
import cv2

# Return annotations
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

def draw_boxes(outpath, name, result, ann):
    image = result.orig_img
    
    for box in result.boxes.xywh:
        x, y, width, height = box.tolist()
        cv2.circle(image, (int(x+width/2), int(y+height/2)), 15, (0, 255, 0), 2)
    if len(ann) > 0:
        for a in ann:
            y, x, z = a.centroid
            cv2.circle(image, (int(x), int(y)), 25, (0, 0, 255), 2)
        
    cv2.imwrite(f'{outpath}/{name}.jpg', image)
    
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

def precision(p_nods, l_nods):
    tp, fp, fn = tp_fp_fn(p_nods, l_nods)
    return tp / (tp + fp)

def recall(p_nods, l_nods):
    tp, fp, fn = tp_fp_fn(p_nods, l_nods)
    return tp / (tp + fn)

def f1(p_nods, l_nods):
    prec = precision(p_nods, l_nods)
    rec = recall(p_nods, l_nods)
    return 2 * (prec * rec) / (prec + rec)

def get_metrics(p_nods, l_nods):
    prec = precision(p_nods, l_nods)
    rec = recall(p_nods, l_nods)
    f1_score = f1(p_nods, l_nods)
    return prec, rec, f1_score