import numpy as np
import pickle
import cv2
import sys
import os
sys.path.append('../')
from utilities import measure_distance, measure_xy_distance


class CameraMovementEstimator:
    def __init__(self, frame):
        self.min_distance = 5
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        first_frame_grey = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grey)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1
        self.features = dict(maxCorners=100,
                             qualityLevel=0.3,
                             minDistance=3,
                             blockSize=7,
                             mask=mask_features)

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_number, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_number]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_number][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_path=None, save_path=None):
        if read_path and os.path.exists(read_path):
            with open(read_path, 'rb') as f:
                camera_movement = pickle.load(f)
        else:
            camera_movement = [[0, 0]] * len(frames)
            old_grey = cv2.cvtColor(src=frames[0], code=cv2.COLOR_BGR2GRAY)
            old_features = cv2.goodFeaturesToTrack(image=old_grey, **self.features)
            for frame_number in range(1, len(frames)):
                frame_grey = cv2.cvtColor(src=frames[frame_number], code=cv2.COLOR_BGR2GRAY)
                new_features, _, _ = cv2.calcOpticalFlowPyrLK(prevImg=old_grey, nextImg=frame_grey, prevPts=old_features, nextPts=None, **self.lk_params)
                max_distance = 0
                camera_movement_x, camera_movement_y = 0, 0
                for i, (new, old) in enumerate(zip(new_features, old_features)):
                    new_features_point = new.ravel()
                    old_features_point = old.ravel()
                    distance = measure_distance(position_1=new_features_point, position_2=old_features_point)
                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x, camera_movement_y = measure_xy_distance(position_1=old_features_point, position_2=new_features_point)
                if max_distance > self.min_distance:
                    camera_movement[frame_number] = [camera_movement_x, camera_movement_y]
                    old_features = cv2.goodFeaturesToTrack(image=frame_grey, **self.features)
                old_grey = frame_grey.copy()
            if save_path:
                with open(save_path, 'wb') as f:
                    pickle.dump(camera_movement, f)
        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_number, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(img=overlay, pt1=(25, 25), pt2=(550, 145), color=(230, 230, 230), thickness=-1)
            alpha = 0.6
            cv2.addWeighted(src1=overlay, alpha=alpha, src2=frame, beta=1 - alpha, gamma=0, dst=frame)
            x_movement, y_movement = camera_movement_per_frame[frame_number]
            cv2.putText(img=frame, text=f'Camera Movement X: {x_movement: .2f}', org=(50, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(25, 25, 25), thickness=3)
            cv2.putText(img=frame, text=f'Camera Movement Y: {y_movement: .2f}', org=(50, 125), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(25, 25, 25), thickness=3)
            output_frames.append(frame)
        return output_frames























