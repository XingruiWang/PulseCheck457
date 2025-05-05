# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import json, os, math
from collections import defaultdict
import pdb
import numpy as np
import ipdb


"""
Utilities for working with function program representations of questions.

Some of the metadata about what question node types are available etc are stored
in a JSON metadata file.
"""


# Handlers for answering questions. Each handler receives the scene structure
# that was output from Blender, the node, and a list of values that were output
# from each of the node's inputs; the handler should return the computed output
# value from this node.


def scene_handler(scene_struct, inputs, side_inputs):
    # Just return all objects in the scene
    return list(range(len(scene_struct['objects'])))


def make_filter_handler(attribute, metadata=None, is_part=False):
    def filter_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1
        assert len(side_inputs) == 1
        value = side_inputs[0]
        output = []
        for idx in inputs[0]:
            if is_part:
                obj_idx, part_idx = [int(a) for a in idx.split('_')]
                try:
                    part = scene_struct['objects'][obj_idx]['_parts'][part_idx]
                except:
                    ipdb.set_trace()
                atr = part[attribute]
            else:
                atr = scene_struct['objects'][idx][attribute]
            if not isinstance(value, str):
                ipdb.set_trace()
            if atr is not None and (value == atr or value in atr):
                output.append(idx)
            elif atr in metadata['_shape_hier']: # shape hier
                if metadata['_shape_hier'][atr] == value:
                    output.append(idx)
        return output
    return filter_handler


def unique_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    if len(inputs[0]) != 1:
        return '__INVALID__'
    return inputs[0][0]


def vg_relate_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 1
    output = set()
    for rel in scene_struct['relationships']:
        if rel['predicate'] == side_inputs[0] and rel['subject_idx'] == inputs[0]:
            output.add(rel['object_idx'])
    return sorted(list(output))



def relate_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 1
    relation = side_inputs[0]
    if relation in ['forward', 'backward']:
        return [obj['obj_id'] for obj in scene_struct['collision_relationships'][relation][inputs[0]]]
    return scene_struct['relationships'][relation][inputs[0]]
        

def union_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return sorted(list(set(inputs[0]) | set(inputs[1])))


def intersect_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return sorted(list(set(inputs[0]) & set(inputs[1])))


def count_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    return len(inputs[0])


def make_same_attr_handler(attribute):
    def same_attr_handler(scene_struct, inputs, side_inputs):
        cache_key = '_same_%s' % attribute
        if cache_key not in scene_struct:
            cache = {}
            for i, obj1 in enumerate(scene_struct['objects']):
                same = []
                for j, obj2 in enumerate(scene_struct['objects']):
                    if i != j and obj1[attribute] == obj2[attribute]:
                        same.append(j)
                cache[i] = same
            scene_struct[cache_key] = cache

        cache = scene_struct[cache_key]
        assert len(inputs) == 1
        assert len(side_inputs) == 0
        return cache[inputs[0]]
    return same_attr_handler


def make_query_handler(attribute, is_part=None):

    def query_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1
        assert len(side_inputs) == 0
        idx = inputs[0]
        if type(idx) == int:
            is_part = False
        else:
            assert '_' in idx
            is_part = True
        if is_part:
            obj_idx, part_idx = [int(a) for a in idx.split('_')]
            part = scene_struct['objects'][obj_idx]['_parts'][part_idx] #part
            obj = part
        else:
            obj = scene_struct['objects'][idx]
        assert attribute in obj
        val = obj[attribute]
        if type(val) == list and len(val) != 1:
            return '__INVALID__'
        elif type(val) == list and len(val) == 1:
            return val[0]
        else:
            return val
    return query_handler

def query_pose_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    idx = inputs[0]
    obj = scene_struct['objects'][idx]
    assert 'pose' in obj
    pose = obj['pose'].split('_pose')[0]
    if pose == 'None':
        return '__INVALID__'
    return pose

def filter_occluder_handler(scene_struct, inputs, side_inputs):
    # output += [int(obj) for obj in scene_struct['_occlusion_relation'][token].get(str(idx), [])]
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    
    output = []
    for idx in inputs[0]:
        if str(idx) in scene_struct['_occlusion_relation']['occluding']:
            output.append(idx)
    output = [int(obj_idx) for obj_idx in scene_struct['_occlusion_relation']['occluding']]
    return output 


def filter_occludee_handler(is_part = True):

    def _partfilter_occludee_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1
        assert len(side_inputs) == 0
        
        output = []
        if is_part:
            for idx in inputs[0]:
                if isinstance(idx, int):
                    ipdb.set_trace()
                obj_idx, part_idx = idx.split('_')
                # [0] is the part name
                if scene_struct['_occlusion'][str(obj_idx)][int(part_idx)][1] > 0:
                    output.append(idx)
            return output
        else:
            # ipdb.set_trace()
            for obj_idx in inputs[0]:
                if scene_struct['occlusion'][str(obj_idx)]['obj'][0] > 0 and len(scene_struct['occlusion'][str(obj_idx)]['obj']) == 4:
                    output.append(int(obj_idx))
            return output 
    return _partfilter_occludee_handler           


def query_occlusion_handler(is_part):

    def _query_occlusion_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1
        assert len(side_inputs) == 0
        obj_idx, part_idx = inputs[0].split('_')
        if is_part:
            occlusion = scene_struct['_occlusion'][str(obj_idx)][int(part_idx)][1] > 0
        else:
            occlusion = scene_struct['occlusion'][str(obj_idx)]['obj'][0] > 0

        return occlusion
    return _query_occlusion_handler

def exist_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    return len(inputs[0]) > 0


def equal_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return inputs[0] == inputs[1]


def less_than_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return inputs[0] < inputs[1]


def greater_than_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 2
    assert len(side_inputs) == 0
    return inputs[0] > inputs[1]


def object2part_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    outputs = []
    if type(inputs[0]) == list:
        objs = inputs[0]
    else:
        assert type(inputs[0]) == int
        objs = [inputs[0]]
    for obj_idx in objs:
        object_parts = scene_struct['objects'][obj_idx]['_parts']
        object_parts = {k:v for k, v in object_parts.items() if v['color'] is not None}
        part_idxs = [k for k in object_parts.keys() if 'color' in object_parts[k]]
        outputs.extend([str(obj_idx)+'_'+str(part_idx) for part_idx in part_idxs])
    return outputs
        

def object2part_all_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    outputs = []
    if type(inputs[0]) == list:
        objs = inputs[0]
    else:
        assert type(inputs[0]) == int
        objs = [inputs[0]]
    for obj_idx in objs:
        part_idxs = list(scene_struct['_occlusion'][str(obj_idx)].keys())
        outputs.extend([str(obj_idx)+'_'+str(part_idx) for part_idx in part_idxs])
    return outputs

def part2object_handler(scene_struct, inputs, side_inputs):
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    outputs = set()
    if type(inputs[0]) == list:
        parts = inputs[0]
    else:
        assert '_' in inputs[0]
        parts = [inputs[0]]
    for idx in parts:
        obj_idx, part_idx = [int(a) for a in idx.split('_')]
        outputs.add(obj_idx)
    outputs = list(outputs)
    return outputs

def pose_relate(pose1, pose2, d1, d2):
    if pose1 == 'None':
        return "None"

    diff = np.abs(d1 - d2)
    diff = np.minimum(diff, 360 - diff)

    assert diff >= 0 and diff <= 180
    if diff < 45:
        return 'same'
    elif diff > 135:
        return 'opposite'
    else:
        return 'vertical'


    # if (np.abs(d1 - d2) % 360 < 45 or np.abs(d1 - d2) % 360 > 315):
    #     return "same"
    # elif (np.abs(d1 + 180 - d2) % 360 < 45 or np.abs(d1 + 180 - d2) % 360 > 315):
    #     return "opposite"
    # elif (np.abs(90 + d1 - d2) % 360 < 45 or \
    #     np.abs(270 + d1 - d2) % 360 < 45):
    #     return "vertical"
    
# def pose_relate(pose1, pose2, d1, d2):
#     if pose1 == 'None' or pose2 == 'None':
#         return "None"
#     if pose1 == pose2 and (np.abs(d1 - d2) % 360 < 45 or np.abs(d1 - d2) % 360 > 315):
#         return "same"
#     elif (pose1, pose2) in [('left_pose', 'right_pose'),
#                           ('right_pose', 'left_pose'),
#                           ('front_pose', 'back_pose'),
#                           ('back_pose', 'front_pose')] and \
#         (np.abs(d1 + 180 - d2) % 360 < 45 or np.abs(d1 + 180 - d2) % 360 > 315):
#         return "opposite"
#     elif ((pose1, pose2) in [('left_pose', 'back_pose'),
#                           ('back_pose', 'right_pose'),
#                           ('right_pose', 'front_pose'),
#                           ('front_pose', 'left_pose')] or \
#         (pose2, pose1) in [('left_pose', 'back_pose'),
#                             ('back_pose', 'right_pose'),
#                             ('right_pose', 'front_pose'),
#                             ('front_pose', 'left_pose')]) and \
#         (np.abs(90 + d1 - d2) % 360 < 45 or \
#         np.abs(270 + d1 - d2) % 360 < 45):
#         return "vertical"
    
def same_pose_handler(scene_struct, inputs, side_inputs):
    cache_key = '_same_pose'
    if cache_key not in scene_struct:
        cache = {}
        for i, obj1 in enumerate(scene_struct['objects']):
            same = []
            for j, obj2 in enumerate(scene_struct['objects']):
                if i != j and pose_relate(obj1['pose'] , obj2['pose'], obj1['pose_degree'] , obj2['pose_degree']) == 'same': 
                    same.append(j)
            cache[i] = same
        scene_struct[cache_key] = cache

    cache = scene_struct[cache_key]
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    return cache[inputs[0]]    


def vertical_pose_handler(scene_struct, inputs, side_inputs):
    cache_key = '_vertical_pose'
    if cache_key not in scene_struct:
        cache = {}
        for i, obj1 in enumerate(scene_struct['objects']):
            same = []
            for j, obj2 in enumerate(scene_struct['objects']):
                if i != j and pose_relate(obj1['pose'] , obj2['pose'], obj1['pose_degree'] , obj2['pose_degree']) == 'vertical': 
                    same.append(j)
            cache[i] = same
        scene_struct[cache_key] = cache

    cache = scene_struct[cache_key]
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    return cache[inputs[0]] 
       
def oppo_pose_handler(scene_struct, inputs, side_inputs):
    cache_key = '_oppo_pose'
    if cache_key not in scene_struct:
        cache = {}
        for i, obj1 in enumerate(scene_struct['objects']):
            same = []
            for j, obj2 in enumerate(scene_struct['objects']):
                if i != j and pose_relate(obj1['pose'] , obj2['pose'], obj1['pose_degree'] , obj2['pose_degree']) == 'opposite': 
                    same.append(j)
            cache[i] = same
        scene_struct[cache_key] = cache

    cache = scene_struct[cache_key]
    assert len(inputs) == 1
    assert len(side_inputs) == 0
    return cache[inputs[0]]    

# def vertical_pose_handler(scene_struct, inputs, side_inputs):
#     cache_key = '_oppo_pose'
#     if cache_key not in scene_struct:
#         cache = {}
#         for i, obj1 in enumerate(scene_struct['objects']):
#             same = []
#             for j, obj2 in enumerate(scene_struct['objects']):
#                 if i != j and (np.abs(90 + obj1['pose_degree'] - obj2['pose_degree']) % 360 < 30 or \
#                              np.abs(270 + obj1['pose_degree'] - obj2['pose_degree']) % 360 < 30): 
#                     same.append(j)
#             cache[i] = same
#         scene_struct[cache_key] = cache

#     cache = scene_struct[cache_key]
#     assert len(inputs) == 1
#     assert len(side_inputs) == 0
#     return cache[inputs[0]]    

def process_pose(rotation):
    if isinstance(rotation, str):
        return rotation
    if rotation >= 0 and rotation < 30:
        pose = 'back_pose'            
    elif rotation > 60 and rotation <= 120:
        pose = 'right_pose' 
    elif rotation > 150 and rotation <= 210:
        pose = 'front_pose'
    elif rotation > 240 and rotation <= 300:
        pose = 'left_pose'        
    elif rotation > 330:
        pose = 'back_pose'   
    else:
        pose = 'None'       
    return pose

def relate_occlusion_handler(token='part_occluded'):
    def _relate_occlusion_handler(scene_struct, inputs, side_inputs):
        assert len(inputs) == 1
        assert len(side_inputs) == 0
        output = []
        # print(inputs[0])
        if token != 'part_occluding':
            idx = inputs[0]
            output += [int(obj) for obj in scene_struct['_occlusion_relation'][token].get(str(idx), [])]
        else:
            idx = inputs[0]
            output += [obj for obj in scene_struct['_occlusion_relation'][token].get(str(idx), [])]
        return output
    return _relate_occlusion_handler

def filter_pose_handler(scene_struct, inputs, side_inputs):
    # print("filter pose")
    # ipdb.set_trace()

    value = side_inputs[0]
    # value = side_inputs[0]
    output = []
    for idx in inputs[0]:
        # rotation = scene_struct['objects'][idx]['rotation']
        pose = scene_struct['objects'][idx]['pose']    
        if value == pose:
            output.append(idx)         
    return output
# Register all of the answering handlers here.
# TODO maybe this would be cleaner with a function decorator that takes
# care of registration? Not sure. Also what if we want to reuse the same engine
# for different sets of node types?
def make_excetue_handlers(key, metadata=None):
    # print(key)
    execute_handlers = {
        'scene': scene_handler,
        'filter_color': make_filter_handler('color', metadata=metadata),
        'filter_shape': make_filter_handler('shape', metadata=metadata),
        'filter_material': make_filter_handler('material', metadata=metadata),
        'filter_size': make_filter_handler('size', metadata=metadata),
        'filter_objectcategory': make_filter_handler('objectcategory', metadata=metadata),
        'partfilter_color': make_filter_handler('color', metadata=metadata, is_part=True),
        'partfilter_partname': make_filter_handler('partname', metadata=metadata, is_part=True),
        'partfilter_material': make_filter_handler('material', metadata=metadata, is_part=True),
        'partfilter_size': make_filter_handler('size', metadata=metadata, is_part=True),
        'unique': unique_handler,
        'relate': relate_handler,
        'union': union_handler,
        'intersect': intersect_handler,
        'count': count_handler,
        'query_color': make_query_handler('color', is_part=False),
        'query_material': make_query_handler('material', is_part=False),
        'query_size': make_query_handler('size', is_part=False),
        'query_shape': make_query_handler('shape', is_part=False),
        'partquery_color': make_query_handler('color', is_part=True),
        'partquery_partname': make_query_handler('partname', is_part=True),
        'partquery_material': make_query_handler('material', is_part=True),
        'partquery_size': make_query_handler('size', is_part=True),
        'exist': exist_handler,
        'equal_color': equal_handler,
        'equal_shape': equal_handler,
        'equal_integer': equal_handler,
        'equal_material': equal_handler,
        'equal_size': equal_handler,
        'equal_object': equal_handler,
        'less_than': less_than_handler,
        'greater_than': greater_than_handler,
        'same_color': make_same_attr_handler('color'),
        'same_shape': make_same_attr_handler('shape'),
        'same_size': make_same_attr_handler('size'),
        'same_material': make_same_attr_handler('material'),
        'same_pose': same_pose_handler,
        'opposite_pose': oppo_pose_handler,
        'vertical_pose': vertical_pose_handler,
        'object2part': object2part_handler,
        'object2part_all': object2part_all_handler,
        'part2object': part2object_handler,
        'query_pose' : query_pose_handler,
        'filter_pose' :filter_pose_handler,
        'partfilter_occludee': filter_occludee_handler(is_part=True),
        'filter_occludee': filter_occludee_handler(is_part=False),
        'filter_occluder': filter_occluder_handler,
        'partquery_occlusion': query_occlusion_handler(is_part=True),
        # occlusion relationship
        'relate_occluding': relate_occlusion_handler(token='occluding'),
        'relate_occluded': relate_occlusion_handler(token='occluded'),
        'relate_occluding_part': relate_occlusion_handler(token='part_occluding'),
        'relate_part_occluded': relate_occlusion_handler(token='part_occluded'),

    }
    return execute_handlers[key]


def answer_question(question, metadata, scene_struct, all_outputs=False,
                                        cache_outputs=True):
    """
    Use structured scene information to answer a structured question. Most of the
    heavy lifting is done by the execute handlers defined above.

    We cache node outputs in the node itself; this gives a nontrivial speedup
    when we want to answer many questions that share nodes on the same scene
    (such as during question-generation DFS). This will NOT work if the same
    nodes are executed on different scenes.
    """
    all_input_types, all_output_types = [], []
    node_outputs = []
    for node in question['nodes']:
        # print('answer question', node)
        if cache_outputs and '_output' in node:
            node_output = node['_output']
        else:
            node_type = node['type']
            # msg = 'Could not find handler for "%s"' % node_type
            # assert node_type in execute_handlers, msg
            # handler = execute_handlers[node_type]
            handler = make_excetue_handlers(node_type, metadata)
            
            node_inputs = [node_outputs[idx] for idx in node['inputs']]
            side_inputs = node.get('side_inputs', [])
            node_output = handler(scene_struct, node_inputs, side_inputs)
            if cache_outputs:
                node['_output'] = node_output
        node_outputs.append(node_output)
        if node_output == '__INVALID__':
            break

    if all_outputs:
        return node_outputs
    else:
        return node_outputs[-1]


def insert_scene_node(nodes, idx):
    # First make a shallow-ish copy of the input
    new_nodes = []
    for node in nodes:
        new_node = {
            'type': node['type'],
            'inputs': node['inputs'],
        }
        if 'side_inputs' in node:
            new_node['side_inputs'] = node['side_inputs']
        new_nodes.append(new_node)

    # Replace the specified index with a scene node
    new_nodes[idx] = {'type': 'scene', 'inputs': []}

    # Search backwards from the last node to see which nodes are actually used
    output_used = [False] * len(new_nodes)
    idxs_to_check = [len(new_nodes) - 1]
    while idxs_to_check:
        cur_idx = idxs_to_check.pop()
        output_used[cur_idx] = True
        idxs_to_check.extend(new_nodes[cur_idx]['inputs'])

    # Iterate through nodes, keeping only those whose output is used;
    # at the same time build up a mapping from old idxs to new idxs
    old_idx_to_new_idx = {}
    new_nodes_trimmed = []
    for old_idx, node in enumerate(new_nodes):
        if output_used[old_idx]:
            new_idx = len(new_nodes_trimmed)
            new_nodes_trimmed.append(node)
            old_idx_to_new_idx[old_idx] = new_idx

    # Finally go through the list of trimmed nodes and change the inputs
    for node in new_nodes_trimmed:
        new_inputs = []
        for old_idx in node['inputs']:
            new_inputs.append(old_idx_to_new_idx[old_idx])
        node['inputs'] = new_inputs

    return new_nodes_trimmed


def is_degenerate(question, metadata, scene_struct, answer=None, verbose=False):
    """
    A question is degenerate if replacing any of its relate nodes with a scene
    node results in a question with the same answer.
    """
    if answer is None:
        answer = answer_question(question, metadata, scene_struct)

    for idx, node in enumerate(question['nodes']):
        if node['type'] == 'relate':
            new_question = {
                'nodes': insert_scene_node(question['nodes'], idx)
            }
            new_answer = answer_question(new_question, metadata, scene_struct)
            if verbose:
                print('here is truncated question:')
                for i, n in enumerate(new_question['nodes']):
                    name = n['type']
                    if 'side_inputs' in n:
                        name = '%s[%s]' % (name, n['side_inputs'][0])
                    # print(i, name, n['_output'])
                print('new answer is: ', new_answer)

            if new_answer == answer:
                return True

    return False

