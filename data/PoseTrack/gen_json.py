import json
import os
from os.path import join, isdir
from os import mkdir, makedirs
import cv2
import numpy as np
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"] """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def natural_sort(given_list):
    """ Sort the given list in the way that humans expect."""
    given_list.sort(key=alphanum_key)


def get_immediate_childfile_paths(folder_path, ext=None, exclude=None):
    files_names = get_immediate_childfile_names(folder_path, ext, exclude)
    files_full_paths = [os.path.join(folder_path, file_name) for file_name in files_names]
    return files_full_paths


def get_immediate_childfile_names(folder_path, ext=None, exclude=None):
    files_names = [file_name for file_name in next(os.walk(folder_path))[2]]
    if ext is not None:
        files_names = [file_name for file_name in files_names
                       if file_name.endswith(ext)]
    if exclude is not None:
        files_names = [file_name for file_name in files_names
                       if not file_name.endswith(exclude)]
    natural_sort(files_names)
    return files_names


def get_immediate_childimages_paths(folder_path):
    files_names = [file_name for file_name in next(os.walk(folder_path))[1]]
    natural_sort(files_names)
    files_full_paths = [os.path.join(folder_path, file_name) for file_name in files_names]
    return files_full_paths


def read_json_from_file(input_path):
    with open(input_path, "r") as read_file:
        python_data = json.load(read_file)
    return python_data

def clip_bbox(bbox, img_shape):
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    if bbox[2] > img_shape[1]:
        bbox[2] = img_shape[1]
    if bbox[3] > img_shape[0]:
        bbox[3] = img_shape[0]
    return bbox

def crop_hwc_coord(bbox, out_sz=511):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    # crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
    # borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return mapping

def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop

def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def crop_like_SiamFC_coord(bbox, exemplar_size=127, context_amount=0.5, search_size=255):
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (search_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # x = crop_hwc1(image, pos_s_2_bbox(target_pos, s_x), search_size, padding)
    return target_pos, s_x

def crop_like_SiamFCx(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return x


def gen_json(json_list, data_subset):
    snippets = dict()
    for js_file in json_list:
        js_data = read_json_from_file(js_file)
        ann = js_data['annotations']
        eg_img_path = join('.', js_data['images'][0]['file_name'])
        im = cv2.imread(eg_img_path)
        im_shape = im.shape
        video_name = js_file.split('.')[-2].split('/')[-2] + '/' + js_file.split('.')[-2].split('/')[-1]
        snippet = dict()


        for i, frame in enumerate(ann):
            #             print(frame)
            if frame['category_id'] != 1:  # 如果标注的不是人
                continue
            if 'bbox' not in frame:  # 如果没有标注bbox(通常是人被完全遮挡，keypoints全为0)
                continue
            kp = frame['keypoints']
            if kp.count(0) >= 30:  # 如果被遮挡的kp数量大于等于10
                continue

            trackid = "{:02d}".format(frame['track_id'])

            frame_name = "{:06d}".format(int(str(frame['image_id'])[-4:]))

            kp_name = "kp_" + frame_name

            bbox = clip_bbox(frame['bbox'], im_shape)

            pos, s = crop_like_SiamFC_coord(bbox, exemplar_size=127, context_amount=0.5, search_size=511)
            mapping_bbox = pos_s_2_bbox(pos, s)
            mapping = crop_hwc_coord(mapping_bbox, out_sz=511)

            affine_bbox = []
            affine_bbox[:2] = affine_transform(bbox[:2], mapping)  # bbox作仿射变换
            affine_bbox[2:] = affine_transform(bbox[2:], mapping)

            joints_3d = np.zeros((int(len(kp) / 3), 3), dtype=np.float)
            for ipt in range(int(len(kp) / 3)):
                joints_3d[ipt, 0] = kp[ipt * 3 + 0]
                joints_3d[ipt, 1] = kp[ipt * 3 + 1]
                joints_3d[ipt, 2] = kp[ipt * 3 + 2]
            pts = joints_3d.copy()
            affine_kp = []
            for j in range(int(len(kp) / 3)):
                if pts[j, 2] > 0:
                    pts[j, :2] = affine_transform(pts[j, :2], mapping)  # kp作仿射变换
                for k in range(3):
                    affine_kp.append(pts[j][k])
            if trackid not in snippet.keys():
                snippet[trackid] = dict()
            #             print("frame_name: ", frame_name)
            #             print("kp_name: ")
            snippet[trackid][frame_name] = affine_bbox
            snippet[trackid][kp_name] = affine_kp

        snippets[video_name] = snippet

    print('save json (dataset), please wait 20 seconds~')
    json.dump(snippets, open('{}_pose_siamfc.json'.format(data_subset), 'w'), indent=4, sort_keys=True)
    print('done!')

def main(instanc_size=511):
    dataDir = '.'
    crop_path = './crop{:d}'.format(instanc_size)
    if not isdir(crop_path): mkdir(crop_path)

    for dataType in ['train', 'val']:
        set_crop_base_path = join(crop_path, dataType)
        set_img_base_path = join(dataDir, 'images', dataType)
        set_ann_base_path = join(dataDir, 'posetrack_data', 'annotations', dataType)

        gt_json_folder_base = "./posetrack_data/annotations/{}".format(dataType)
        gt_json_file_paths = get_immediate_childfile_paths(gt_json_folder_base, ext=".json")
        gt_img_file_paths = get_immediate_childimages_paths(set_img_base_path)

        gt_json_file_video_names = []
        for gt_json_file_path in gt_json_file_paths:
            gt_json_file_video_names.append(os.path.basename(gt_json_file_path).split('.')[0])
        #     print(gt_json_file_video_names)
        #     print(len(gt_json_file_video_names))

        gt_img_file_video_names = []
        for gt_img_file_path in gt_img_file_paths:
            gt_img_file_video_names.append(os.path.basename(gt_img_file_path))
        #     print(gt_img_file_video_names)
        #     print(len(gt_img_file_video_names))

        #   PoseTrack数据集一个标注文件对应一段视频，但标注文件的数量与视频数量不一致，只选择有标注的视频进行crop和gen_json
        gt_img_with_anno_names = [x for x in gt_json_file_video_names if x in gt_img_file_video_names]
        #     print(gt_img_with_anno_names)
        json_list = []
        for js in gt_img_with_anno_names:
            json_list.append(join(gt_json_folder_base, js + '.json'))
        #     print(json_list)
        #     print(json_list[0].split('.')[-2].split('/')[-2] + '/' + json_list[0].split('.')[-2].split('/')[-1])
        # print(len(gt_img_with_anno_names), len(json_list))

        # n_video = len(gt_img_with_anno_names)

        gen_json(json_list, dataType)


if __name__ == '__main__':
    instanc_size = 511
    main(instanc_size)
