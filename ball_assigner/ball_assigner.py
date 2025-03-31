import sys
sys.path.append('../')
from utilities import get_bbox_centre, measure_distance


class BallAssigner:
    def __init__(self):
        self.max_distance = 70

    def assign_ball(self, players, ball_bbox):
        ball_position = get_bbox_centre(ball_bbox)
        min_distance = 99999
        assigned_player = -1
        for player_id, player in players.items():
            player_bbox = player['bbox']
            distance_left = measure_distance(position_1=(player_bbox[0], player_bbox[-1]), position_2=ball_position)
            distance_right = measure_distance(position_1=(player_bbox[2], player_bbox[-1]), position_2=ball_position)
            distance = min(distance_left, distance_right)
            if distance < self.max_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id
        return assigned_player
