import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# monkey patch
if not hasattr(np, 'int'):
    print("monkey patch np.int to np.int32")
    np.int = np.int32  

# ... existing imports and setup code ...

def calculate_head_movement(faces,img_shape):
    """
    calculate the movement score of the head
    by using the distance between the keypoints of two frames
    """
    if len(faces) < 2:
        return {
        'avg_movement': 0, 
        'max_movement': 0,
        'min_movement_id': 0,
    }
    
    movements = []
    
    img_width,img_height = img_shape[1],img_shape[0]

    for i in range(len(faces)-1):
        curr_kps = faces[i].kps
        next_kps = faces[i+1].kps
        movement = np.mean(np.abs(curr_kps - next_kps)/min(img_width,img_height))
        movements.append(movement)
    
    avg_movement = float(np.mean(movements))
    max_movement = float(np.max(movements))
    min_movement_id = int(np.argmax(movements))

    return {
        'avg_movement': avg_movement, 
        'max_movement': max_movement,
        'min_movement_id': min_movement_id
    }

def calculate_face_rotation(faces):
    """
    calculate the rotation score of the face, and using
    the average rotation amplitude between adjacent faces from three angles
    """
    if len(faces) < 2:
        return {    
            'avg_rotation': 0.0,
            'min_rotation': 0.0,
            'min_rotation_id':0  
        }
    
    rotation_amplitudes = []
    for i in range(len(faces)-1):
        curr_face = faces[i]
        next_face = faces[i+1]
        
        # current and next face pose
        curr_pitch, curr_yaw, curr_roll = curr_face.pose
        next_pitch, next_yaw, next_roll = next_face.pose
        
        # ajacent face rotation amplitudes
        pitch_amplitude = abs(curr_pitch - next_pitch)
        yaw_amplitude = abs(curr_yaw - next_yaw)
        roll_amplitude = abs(curr_roll - next_roll)
        
        # average rotation amplitude
        avg_amplitude = (pitch_amplitude**2 + yaw_amplitude**2 + roll_amplitude**2)**0.5
        
        rotation_amplitudes.append(avg_amplitude)
    
    # score of average rotation amplitude (0-100)
    avg_rotation_amplitude = float(np.mean(rotation_amplitudes))
    max_rotation_amplitude = float(np.max(rotation_amplitudes))
    min_rotation_id = int(np.argmax(rotation_amplitudes))

    return {
        'avg_rotation': avg_rotation_amplitude,
        'min_rotation': max_rotation_amplitude,
        'min_rotation_id':min_rotation_id,
    }


def check_face_completeness(face, img_shape):
    """
    using the face keypoints to check the completeness of the face
    """
    kps = face.kps  
    # print(f"kps: {kps}")
    # breakpoint()
    img_height, img_width = img_shape[:2]
    
    # important face regions and their weights
    key_points_groups = {
        'eyes': {'points': [0, 1], 'weight': 0.3},  
        'nose': {'points': [2], 'weight': 0.4},     
        'mouth': {'points': [3, 4], 'weight': 0.3}, 
    }
    
    total_score = 0
    
    for group_name, group_info in key_points_groups.items():
        group_points = group_info['points']
        weight = group_info['weight']
        
        # check if the points are in the image
        valid_points = 0
        for point_idx in group_points:
            x, y = kps[point_idx]
            # print(f'x: {x}, img_width: {img_width}, y: {y}, img_height: {img_height}')
            if (0 <= x < img_width and 0 <= y < img_height):
                valid_points += 1
        
        group_score = (valid_points / len(group_points)) * 100
        total_score += group_score * weight
    
    # breakpoint()
    return float(total_score) # change from 100 scale to float scale




def calculate_face_resolution_score(face, img_shape):
    """
    calculate the resolution score of the face based on the proportion of face area to total image area
    """
    bbox = face.bbox
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    
    face_area = face_width * face_height
    total_area = img_shape[0] * img_shape[1]
    
    proportion = face_area / total_area
    
    score = proportion * 100
    
    return float(score)

def calculate_head_orientation_score(face):
    """
    uses three angles to evaluate the head orientation, 
    using the square root of the sum of the squares of the three angles
    """
    pitch, yaw, roll = face.pose

    pitch_score = abs(pitch) / 180 * 100
    yaw_score = abs(yaw) / 180 * 100
    roll_score = abs(roll) / 180 * 100
    
    return float(pitch_score**2 + yaw_score**2 + roll_score**2)**0.5

if __name__ == "__main__":
    print('running testing code in pic_head_base_fun.py')
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640,640))
    # test_face.jpg
    # img = cv2.imread("./src/half_face.png")
    # img = cv2.imread("./src/upper_half_face.png")

    img = ins_get_image('t1')
    faces = app.get(img)
    # breakpoint()
    # completeness_score = check_face_completeness(faces[0], img.shape)
    # print(f"completeness score: {completeness_score:.2f}")


    # calulate head movement and face rotation
    movement_dic=calculate_head_movement(faces,img.shape)
    avg_head_movement_score = movement_dic['avg_movement']
    max_head_movement_score = movement_dic['max_movement']
    
    face_rotation_dic = calculate_face_rotation(faces)
    avg_face_rotation_score = face_rotation_dic['avg_rotation']
    max_face_rotation_score = face_rotation_dic['max_rotation']

    print(f"avg head movement score: {avg_head_movement_score:.2f}")
    print(f"max head movement score: {max_head_movement_score:.2f}")
    print(f"avg face rotation score: {avg_face_rotation_score:.2f}")
    print(f"max face rotation score: {max_face_rotation_score:.2f}")

    for i, face in enumerate(faces):
        completeness_score = check_face_completeness(face, img.shape)
        resolution_score = calculate_face_resolution_score(face,img.shape)# TODO, use proportion
        orientation_score = calculate_head_orientation_score(face)
        
        print(f"\nface #{i+1} score:")
        print(f"face completeness score: {completeness_score:.2f}")
        print(f"face resolution score: {resolution_score:.2f}")
        print(f"head orientation score: {orientation_score:.2f}")


