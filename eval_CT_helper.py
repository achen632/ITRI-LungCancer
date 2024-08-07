import pylidc as pl
import cv2

# Return annotations
def get_ann_from_slice(patient_id, slice_location):
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
        x_center, y_center, width, height = box.tolist()

        # x1 = int(x_center - width / 2)
        # y1 = int(y_center - height / 2)
        # x2 = int(x_center + width / 2)
        # y2 = int(y_center + height / 2)
        
        cv2.circle(image, (int(x_center), int(y_center)), 15, (0, 255, 0), 2)
        
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    if len(ann) > 0:
        # x_all = []
        # y_all = []
        for a in ann:
            x, y, z = a.centroid
            # x_all.append(int(x))
            # y_all.append(int(y))
            cv2.circle(image, (int(y), int(x)), 25, (0, 0, 255), 2)
        # cv2.circle(image, (int(np.mean(y_all)), int(np.mean(x_all))), 15, (0, 0, 255), 2)
        
    cv2.imwrite(f'{outpath}/{name}.jpg', image)
    
# Returns bounding box from result
def getBoundingBox(result):
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
    first = get_first_slice(index-1, boxes, ref)
    last = get_last_slice(index+1, boxes, ref)
    return last-first+1

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