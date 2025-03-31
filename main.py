from speed_and_distance_estimator import SpeedAndDistanceEstimator
from camera_movement_estimator import CameraMovementEstimator
from utilities import read_video, save_video
from view_transformer import ViewTransformer
from team_assigner import TeamAssigner
from ball_assigner import BallAssigner
from tracker import Tracker
import numpy as np
import os


def main(save=True):
    # read video
    frames = read_video(input_path='input/input.mp4')

    # initialize tracker
    tracker = Tracker(model_path='models/best.pt')
    tracks = tracker.get_object_tracks(frames=frames, read_path='saved_data/tracks.pkl', save_path='saved_data/tracks.pkl')

    # get object positions
    tracker.add_position_to_tracks(tracks=tracks)

    # estimate camera movement
    camera_movement_estimator = CameraMovementEstimator(frame=frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames, read_path='saved_data/camera_movement.pkl', save_path='saved_data/camera_movement.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks=tracks, camera_movement_per_frame=camera_movement_per_frame)

    # add view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks=tracks)

    # interpolate ball position
    tracks['ball'] = tracker.interpolate_ball_positions(ball_positions=tracks['ball'])

    # estimate speed and distance
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks=tracks)

    # assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colour(frame=frames[0], player_detections=tracks['players'][0])
    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frame=frames[frame_number], player_bbox=track['bbox'], player_id=player_id)
            tracks['players'][frame_number][player_id]['team'] = team
            tracks['players'][frame_number][player_id]['team_colour'] = team_assigner.team_colours[team]

    # assign ball possession
    ball_assigner = BallAssigner()
    team_ball_possession = []
    for frame_number, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_number][1]['bbox']
        assigned_player = ball_assigner.assign_ball(players=player_track, ball_bbox=ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_number][assigned_player]['in_possession'] = True
            team_ball_possession.append(tracks['players'][frame_number][assigned_player]['team'])
        else:
            team_ball_possession.append(team_ball_possession[-1])
    team_ball_possession = np.array(team_ball_possession)

    # draw output
    output_frames = tracker.draw_annotations(frames=frames, tracks=tracks, team_ball_possession=team_ball_possession)

    # draw camera movement
    output_frames = camera_movement_estimator.draw_camera_movement(frames=output_frames, camera_movement_per_frame=camera_movement_per_frame)

    # draw speed and distance
    output_frames = speed_and_distance_estimator.draw_speed_and_distance(frames=output_frames, tracks=tracks)

    # save video
    if save:
        output_path = f'output/output_{len(os.listdir('output')) + 1}.avi'
        save_video(frames=output_frames, output_path=output_path)
        print(f'Output saved to {output_path}...')


if __name__ == '__main__':
    main(save=True)
