import cv2
import numpy as np
import os
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
import json
import glob
import re
import matplotlib.pyplot as plt


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


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_video(vid_id, gt_json_file_path, crop_path, gt_img_folder_base, instanc_size):
    video_crop_base_path = join(crop_path, vid_id)
    if not isdir(video_crop_base_path): makedirs(video_crop_base_path)

    gt_python_data = read_json_from_file(gt_json_file_path)
    ann = gt_python_data['annotations']
    for obj in ann:
        if obj['category_id'] != 1:
            continue
        if 'bbox' not in obj:
            continue
        kp = obj['keypoints']
        if kp.count(0) >= 30:
            continue
        img_name = '00' + str(obj['image_id'])[-4:] + '.jpg'
        img_path = join(gt_img_folder_base, vid_id, img_name)
        im = cv2.imread(img_path)
        avg_chans = np.mean(im, axis=(0, 1))
        bbox = obj['bbox']
        bbox = clip_bbox(bbox, im.shape)
        trackid = obj['track_id']
        x = crop_like_SiamFCx(im, bbox, instanc_size=instanc_size, padding=avg_chans)
        cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(str(obj['image_id'])[-4:]), trackid)), x)


def main(instanc_size=511, num_threads=12):
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
        print(len(gt_img_with_anno_names), len(json_list))

        n_video = len(gt_img_with_anno_names)

        #     gen_json(json_list, dataType)
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, json_path, join(gt_json_folder_base, json_path + '.json'),
                                  set_crop_base_path, set_img_base_path, instanc_size)
                  for json_path in gt_img_with_anno_names]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_video, prefix=dataType, suffix='Done ', barLength=40)

    print('done!')


if __name__ == '__main__':
    since = time.time()
    main(int(sys.argv[1]), int(sys.argv[2]))
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
