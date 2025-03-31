import numpy as np
import cv2


class ViewTransformer:
    def __init__(self):
        self.pitch_width = 68
        self.pitch_length = 23.32
        self.pixel_vertexes = np.array([[110, 1035], [265, 275], [910, 260], [1640, 915]])
        self.pixel_vertexes = self.pixel_vertexes.astype(np.float32)
        self.target_vertexes = np.array([[0, self.pitch_width], [0, 0], [self.pitch_length, 0], [self.pitch_length, self.pitch_width]])
        self.target_vertexes = self.target_vertexes.astype(np.float32)
        self.perspective_transformer = cv2.getPerspectiveTransform(src=self.pixel_vertexes, dst=self.target_vertexes)

    def transform_position(self, position):
        is_inside = cv2.pointPolygonTest(contour=self.pixel_vertexes, pt=(int(position[0]), int(position[1])), measureDist=False) >= 0
        if not is_inside:
            return None
        reshaped_position = position.reshape(-1, 1, 2).astype(np.float32)
        transformed_position = cv2.perspectiveTransform(src=reshaped_position, m=self.perspective_transformer)
        return transformed_position.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_number, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    transformed_position = self.transform_position(position=position)
                    if transformed_position is not None:
                        transformed_position = transformed_position.squeeze().tolist()
                    tracks[object][frame_number][track_id]['position_transformed'] = transformed_position
