import cv2
import sys
sys.path.append('../')
from utilities import measure_distance, get_foot_position


class SpeedAndDistanceEstimator:
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        for object, object_tracks in tracks.items():
            if object in ['ball', 'referee']:
                continue
            number_of_frames = len(object_tracks)
            for frame_number in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_number + self.frame_window, number_of_frames - 1)
                for track_id, _ in object_tracks[frame_number].items():
                    if track_id not in object_tracks[last_frame]:
                        continue
                    start_position = object_tracks[frame_number][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']
                    if not start_position or not end_position:
                        continue
                    distance_covered = measure_distance(position_1=start_position, position_2=end_position)
                    time_elapsed = (last_frame - frame_number) / self.frame_rate
                    speed_mps = distance_covered / time_elapsed
                    speed_kph = speed_mps * 3.6
                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    total_distance[object][track_id] += distance_covered
                    for frame_number_batch in range(frame_number, last_frame):
                        if track_id not in tracks[object][frame_number_batch]:
                            continue
                        tracks[object][frame_number_batch][track_id]['speed'] = speed_kph
                        tracks[object][frame_number_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_number, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object in ['ball', 'referee']:
                    continue
                for _, track_info in object_tracks[frame_number].items():
                    if 'speed' in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if not speed or not distance:
                            continue
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox=bbox)
                        position = list(position)
                        position[1] += 40
                        position = tuple(map(int, position))
                        cv2.putText(img=frame, text=f'{speed:.2f} km/h', org=position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(25, 25, 25), thickness=2)
                        cv2.putText(img=frame, text=f'{distance:.2f} m', org=(position[0], position[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(25, 25, 25), thickness=2)
            output_frames.append(frame)
        return output_frames
