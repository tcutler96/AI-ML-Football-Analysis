from ultralytics import YOLO
import supervision as sv
import pandas as pd
import numpy as np
import pickle
import cv2
import sys
import os
sys.path.append('../')
from utilities import get_bbox_centre, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_number, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = get_ = track_info['bbox']
                    if object == 'ball':
                        position = get_bbox_centre(bbox=bbox)
                    else:
                        position = get_foot_position(bbox=bbox)
                    tracks[object][frame_number][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detection = self.model.predict(frames[i: i + batch_size], conf=0.1)
            detections += detection
        return detections

    def get_object_tracks(self, frames, read_path=None, save_path=None):
        if read_path and os.path.exists(read_path):
            with open(read_path, 'rb') as f:
                tracks = pickle.load(f)
        else:
            detections = self.detect_frames(frames)
            tracks = {'players': [],
                      'referees': [],
                      'ball': []}
            for frame_num, detection in enumerate(detections):
                class_names = detection.names
                class_names_inv = {v: k for k, v in class_names.items()}
                detection_supervision = sv.Detections.from_ultralytics(detection)
                for object_index, class_id in enumerate(detection_supervision.class_id):
                    if class_names[class_id] == 'goalkeeper':
                        detection_supervision.class_id[object_index] = class_names_inv['player']
                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
                tracks['players'].append({})
                tracks['referees'].append({})
                tracks['ball'].append({})
                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    class_id = frame_detection[3]
                    track_id = frame_detection[4]
                    if class_id == class_names_inv['player']:
                        tracks['players'][frame_num][track_id] = {'bbox': bbox}
                    if class_id == class_names_inv['referee']:
                        tracks['referees'][frame_num][track_id] = {'bbox': bbox}
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    class_id = frame_detection[3]
                    if class_id == class_names_inv['ball']:
                        tracks['ball'][frame_num][1] = {'bbox': bbox}
            if save_path:
                with open(save_path, 'wb') as f:
                    pickle.dump(tracks, f)
        return tracks

    def draw_ellipse(self, frame, bbox, colour, track_id=None):
        y2 = int(bbox[3])
        x_centre, _ = get_bbox_centre(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(img=frame, center=(x_centre, y2), axes=(int(width), int(0.35 * width)), angle=0.0, startAngle=-45, endAngle=235, color=colour, thickness=2, lineType=cv2.LINE_4)
        if track_id:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = int(x_centre - rectangle_width // 2)
            x2_rect = int(x_centre + rectangle_width // 2)
            y1_rect = int((y2 - rectangle_height // 2) + 15)
            y2_rect = int((y2 + rectangle_height // 2) + 15)
            cv2.rectangle(img=frame, pt1=(x1_rect, y1_rect), pt2=(x2_rect, y2_rect), color=colour, thickness=cv2.FILLED)
            x1_text = x1_rect + 13
            if track_id > 9:
                x1_text -= 6
            if track_id > 99:
                x1_text -= 6
            cv2.putText(img=frame, text=str(track_id), org=(x1_text, y1_rect + 16), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255 - colour[0], 255 - colour[1], 255 - colour[2]), thickness=2)

        return frame

    def draw_triangle(self, frame, bbox, colour):
        y = int(bbox[1])
        x, _ = get_bbox_centre(bbox)
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(image=frame, contours=[triangle_points], contourIdx=0, color=colour, thickness=cv2.FILLED)
        cv2.drawContours(image=frame, contours=[triangle_points], contourIdx=0, color=(25, 25, 25), thickness=1)
        return frame

    def draw_team_ball_possession(self, frame, frame_number, team_ball_possession):
        overlay = frame.copy()
        cv2.rectangle(img=frame, pt1=(1375, 850), pt2=(1900, 970), color=(230, 230, 230), thickness=cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(src1=overlay, alpha=alpha, src2=frame, beta=1 - alpha, gamma=0, dst=frame)
        team_ball_possession_till_frame = team_ball_possession[:frame_number + 1]
        team_1_possession = team_ball_possession_till_frame[team_ball_possession_till_frame == 1].shape[0]
        team_2_possession = team_ball_possession_till_frame[team_ball_possession_till_frame == 2].shape[0]
        team_1 = team_1_possession / (team_1_possession + team_2_possession)
        team_2 = team_2_possession / (team_1_possession + team_2_possession)
        cv2.putText(img=frame, text=f'Team 1 Possession: {team_1 * 100: .2f}%', org=(1400, 900), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(25, 25, 25), thickness=3)
        cv2.putText(img=frame, text=f'Team 2 Possession: {team_2 * 100: .2f}%', org=(1400, 950), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(25, 25, 25), thickness=3)
        return frame

    def draw_annotations(self, frames, tracks, team_ball_possession):
        output_frames = []
        for frame_number, frame in enumerate(frames):
            frame = frame.copy()
            player_dict = tracks['players'][frame_number]
            referee_dict = tracks['referees'][frame_number]
            ball_dict = tracks['ball'][frame_number]
            for track_id, player in player_dict.items():
                colour = player.get('team_colour', (210, 30, 140))
                frame = self.draw_ellipse(frame=frame, bbox=player['bbox'], colour=colour, track_id=track_id)
                if player.get('in_possession', False):
                    frame = self.draw_triangle(frame=frame, bbox=player['bbox'], colour=colour)
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame=frame, bbox=referee['bbox'], colour=(25, 25, 25))
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame=frame, bbox=ball['bbox'], colour=(50, 180, 250))
            frame = self.draw_team_ball_possession(frame=frame, frame_number=frame_number, team_ball_possession=team_ball_possession)
            output_frames.append(frame)
        return output_frames
