import json
import os


clevr_scene = json.load(open('/home/xingrui/vqa/super-clevr-gen/output/ver_mask_new/superCLEVR_scenes_210k.json'))


all_scene_paths = os.listdir('/home/xingrui/vqa/super-clevr-gen/output/ver_mask_new/scenes')
all_scene_paths_camera = '/home/xingrui/vqa/super-clevr-gen/output/ver_mask_new-2/scenes-camera'


all_scenes = []
for scene_path in sorted(all_scene_paths):
    ori_scene_path = os.path.join('/home/xingrui/vqa/super-clevr-gen/output/ver_mask_new/scenes', scene_path)
    with open(ori_scene_path, 'r') as f:
        this_scene = json.load(f)

    scene_path_camera = os.path.join(all_scene_paths_camera, scene_path)
    with open(scene_path_camera, 'r') as f:
        this_scene_camera = json.load(f)  

    this_scene['matrix_world'] = this_scene_camera['matrix_world']
    this_scene['matrix_world_inverted'] = this_scene_camera['matrix_world_inverted']
    this_scene['projection_matrix'] = this_scene_camera['projection_matrix']
    

    with open(os.path.join('/home/xingrui/vqa/super-clevr-gen/output/ver_mask_new/scenes-camera', scene_path), 'w') as f:
        json.dump(this_scene, f)
    

    all_scenes.append(this_scene)

clevr_scene['scenes'] = all_scenes

with open('/home/xingrui/vqa/super-clevr-gen/output/ver_mask_new-2/superCLEVR_scenes_210k.json', 'w') as f:
    json.dump(clevr_scene, f)
    



