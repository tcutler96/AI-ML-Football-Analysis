
def get_bbox_centre(bbox):
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


def get_bbox_width(bbox):
    return bbox[2] - bbox[0]


def measure_distance(position_1, position_2):
    return ((position_1[0] - position_2[0]) ** 2 + (position_1[1] - position_2[1]) ** 2) ** 0.5


def measure_xy_distance(position_1, position_2):
    return position_1[0] - position_2[0], position_1[1] - position_2[1]


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
