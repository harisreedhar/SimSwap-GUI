import os
import cv2
import glob
import uuid
import time
import shutil
import subprocess
import platform
import threading
import webbrowser
import numpy as np
import tkinter as tk
from tqdm import tqdm
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageOps, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop

from moviepy.editor import AudioFileClip, VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
from util.add_watermark import watermark_image
from util.reverse2original import encode_segmentation_rgb, SoftErosion, postprocess

# pip install tkinterdnd2
from tkinterdnd2 import DND_FILES, TkinterDnD

######################################## DEFAULTS ########################################
CWD = os.path.curdir
OUTPUT_PATH = os.path.join(CWD, 'output', 'out.mp4')
ARC_PATH = os.path.join(CWD, 'arcface_model', 'arcface_checkpoint.tar')
TEMP_PATH = os.path.join(CWD, 'temp_results', 'sequence')
TRIM_PATH = os.path.join(CWD, 'temp_results', 'trims')
DET_THRESHOLD = 0.6
ICON_SIZE = (100, 100)
WINDOW_SIZE = (800, 450)
CROP_SIZE = 224
MASK_KERNEL = "40 40"
DET_SIZE = 640
DET_THRESHOLD = 0.6
USE_MASK = True
WATERMARK = False
THEME_STYLE = 'clam'
FFHQ = False
FACE_PART_IDS = "1 2 3 4 5 6 10 12 13"
######################################## DEFAULTS ########################################

# The codes are currently a huge mess.
# I don't have too much time to beautify code... So here is what it is.
# will clean things later

class SimSwap:
    def _totensor(array):
        tensor = torch.from_numpy(array)
        img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255)

    def encode_segmentation_rgb(segmentation, no_neck=True, face_part_ids=[1, 2, 3, 4, 5, 6, 10, 12, 13]):
        parse = segmentation
        #face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
        mouth_id = 11
        face_map = np.zeros([parse.shape[0], parse.shape[1]])
        mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
        for valid_id in face_part_ids:
            valid_index = np.where(parse==valid_id)
            face_map[valid_index] = 255
        valid_index = np.where(parse==mouth_id)
        mouth_map[valid_index] = 255
        return np.stack([face_map, mouth_map], axis=2)

    def reverse2wholeimage(b_align_crop_tenor_list,swaped_imgs, mats, crop_size, oriimg, logoclass, save_path = '', \
                        no_simswaplogo = False,pasring_model =None,norm = None, use_mask = False, _kernel_size=(40,40)
                        , face_part_ids=[1, 2, 3, 4, 5, 6, 10, 12, 13]):
        target_image_list = []
        img_mask_list = []
        if use_mask:
            smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()
        else:
            pass
        for swaped_img, mat ,source_img in zip(swaped_imgs, mats,b_align_crop_tenor_list):
            swaped_img = swaped_img.cpu().detach().numpy().transpose((1, 2, 0))
            img_white = np.full((crop_size,crop_size), 255, dtype=float)
            mat_rev = np.zeros([2,3])
            div1 = mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
            mat_rev[0][0] = mat[1][1]/div1
            mat_rev[0][1] = -mat[0][1]/div1
            mat_rev[0][2] = -(mat[0][2]*mat[1][1]-mat[0][1]*mat[1][2])/div1
            div2 = mat[0][1]*mat[1][0]-mat[0][0]*mat[1][1]
            mat_rev[1][0] = mat[1][0]/div2
            mat_rev[1][1] = -mat[0][0]/div2
            mat_rev[1][2] = -(mat[0][2]*mat[1][0]-mat[0][0]*mat[1][2])/div2
            orisize = (oriimg.shape[1], oriimg.shape[0])
            if use_mask:
                source_img_norm = norm(source_img)
                source_img_512  = F.interpolate(source_img_norm,size=(512,512))
                out = pasring_model(source_img_512)[0]
                parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
                vis_parsing_anno = parsing.copy().astype(np.uint8)
                tgt_mask = SimSwap.encode_segmentation_rgb(vis_parsing_anno, face_part_ids=face_part_ids)
                if tgt_mask.sum() >= 5000:
                    target_mask = cv2.resize(tgt_mask, (crop_size,  crop_size))
                    target_image_parsing = postprocess(swaped_img, source_img[0].cpu().detach().numpy().transpose((1, 2, 0)), target_mask,smooth_mask)
                    target_image = cv2.warpAffine(target_image_parsing, mat_rev, orisize)
                else:
                    target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)[..., ::-1]
            else:
                target_image = cv2.warpAffine(swaped_img, mat_rev, orisize)
            img_white = cv2.warpAffine(img_white, mat_rev, orisize)
            img_white[img_white>20] =255
            img_mask = img_white
            kernel = np.ones(_kernel_size,np.uint8)
            img_mask = cv2.erode(img_mask,kernel,iterations = 1)
            kernel_size = (20, 20)
            blur_size = tuple(2*i+1 for i in kernel_size)
            img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
            img_mask /= 255
            img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
            if use_mask:
                target_image = np.array(target_image, dtype=np.float) * 255
            else:
                target_image = np.array(target_image, dtype=np.float)[..., ::-1] * 255
            img_mask_list.append(img_mask)
            target_image_list.append(target_image)
        img = np.array(oriimg, dtype=np.float)
        for img_mask, target_image in zip(img_mask_list, target_image_list):
            img = img_mask * target_image + (1-img_mask) * img

        final_img = img.astype(np.uint8)
        if not no_simswaplogo:
            final_img = logoclass.apply_frames(final_img)
        cv2.imwrite(save_path, final_img)
        return final_img

    def video_swap(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir='./temp_results',
        crop_size=224, no_simswaplogo = False,use_mask =False, self_obj=None, _kernel_size=(40, 40), face_part_ids=[1,2]):
        video_forcheck = VideoFileClip(video_path)
        if video_forcheck.audio is None:
            no_audio = True
        else:
            no_audio = False
        del video_forcheck
        if not no_audio:
            video_audio_clip = AudioFileClip(video_path)
        video = cv2.VideoCapture(video_path)
        logoclass = watermark_image('./simswaplogo/simswaplogo.png')
        ret = True
        frame_index = 0
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        if  os.path.exists(temp_results_dir):
                shutil.rmtree(temp_results_dir)
        spNorm =SpecificNorm()
        if use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net =None
        for frame_index in tqdm(range(frame_count)):
            if self_obj.stop_process:
                return
            ret, frame = video.read()
            if  ret:
                detect_results = detect_model.get(frame,crop_size)
                if detect_results is not None:
                    if not os.path.exists(temp_results_dir):
                            os.mkdir(temp_results_dir)
                    frame_align_crop_list = detect_results[0]
                    frame_mat_list = detect_results[1]
                    swap_result_list = []
                    frame_align_crop_tenor_list = []
                    for frame_align_crop in frame_align_crop_list:
                        frame_align_crop_tenor = SimSwap._totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
                        swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                        cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                        swap_result_list.append(swap_result)
                        frame_align_crop_tenor_list.append(frame_align_crop_tenor)
                    final_img = SimSwap.reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
                        os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,
                        use_mask=use_mask, norm = spNorm, _kernel_size=_kernel_size, face_part_ids=face_part_ids)

                    ########################################### Two pass #####################################################################################
                    if self_obj.settings.two_pass.get():
                        self_obj.set_status(f"Performing two pass... ({int((frame_index/frame_count)*100)}% completed)")
                        detect_results = detect_model.get(final_img,crop_size)
                        if detect_results is not None:
                            if not os.path.exists(temp_results_dir):
                                    os.mkdir(temp_results_dir)
                            frame_align_crop_list = detect_results[0]
                            frame_mat_list = detect_results[1]
                            swap_result_list = []
                            frame_align_crop_tenor_list = []
                            for frame_align_crop in frame_align_crop_list:
                                frame_align_crop_tenor = SimSwap._totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
                                swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                                swap_result_list.append(swap_result)
                                frame_align_crop_tenor_list.append(frame_align_crop_tenor)
                            final_img = SimSwap.reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
                                os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,
                                use_mask=use_mask, norm = spNorm, _kernel_size=_kernel_size, face_part_ids=face_part_ids)
                    ########################################### Two pass #####################################################################################

                    try:
                        if self_obj.stop_process:
                            return
                        self_obj.video_player.update_display(image=final_img)
                        self_obj.set_status(f"Processing {frame_index} of {frame_count}... ({int((frame_index/frame_count)*100)}% completed)")
                    except: pass
                else:
                    if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                    frame = frame.astype(np.uint8)
                    if not no_simswaplogo:
                        frame = logoclass.apply_frames(frame)
                    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                    self_obj.set_status(f"Skipping no face...({int((frame_index/frame_count)*100)}% completed)")
            else:
                break
        video.release()
        self_obj.set_status(f"Merging sequence...")
        path = os.path.join(temp_results_dir,'*.jpg')
        image_filenames = sorted(glob.glob(path))
        clips = ImageSequenceClip(image_filenames,fps = fps)
        if not no_audio:
            clips = clips.set_audio(video_audio_clip)
        clips.write_videofile(save_path,audio_codec='aac')

    def video_target_swap(video_path, id_vetor,specific_person_id_nonorm,id_thres, swap_model,
        detect_model, save_path, temp_results_dir='./temp_results',
        crop_size=224, no_simswaplogo = False,use_mask =False, self_obj=None, _kernel_size=(40, 40), face_part_ids=[1,2]):

        video_forcheck = VideoFileClip(video_path)
        if video_forcheck.audio is None:
            no_audio = True
        else:
            no_audio = False
        del video_forcheck
        if not no_audio:
            video_audio_clip = AudioFileClip(video_path)
        video = cv2.VideoCapture(video_path)
        logoclass = watermark_image('./simswaplogo/simswaplogo.png')
        ret = True
        frame_index = 0
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        if  os.path.exists(temp_results_dir):
                shutil.rmtree(temp_results_dir)
        spNorm =SpecificNorm()
        mse = torch.nn.MSELoss().cuda()
        if use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net =None
        for frame_index in tqdm(range(frame_count)):
            if self_obj.stop_process:
                return
            ret, frame = video.read()
            if  ret:
                detect_results = detect_model.get(frame,crop_size)
                if detect_results is not None:
                    if not os.path.exists(temp_results_dir):
                            os.mkdir(temp_results_dir)
                    frame_align_crop_list = detect_results[0]
                    frame_mat_list = detect_results[1]
                    id_compare_values = []
                    frame_align_crop_tenor_list = []
                    for frame_align_crop in frame_align_crop_list:
                        frame_align_crop_tenor = SimSwap._totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
                        frame_align_crop_tenor_arcnorm = spNorm(frame_align_crop_tenor)
                        frame_align_crop_tenor_arcnorm_downsample = F.interpolate(frame_align_crop_tenor_arcnorm, size=(112,112))
                        frame_align_crop_crop_id_nonorm = swap_model.netArc(frame_align_crop_tenor_arcnorm_downsample)
                        id_compare_values.append(mse(frame_align_crop_crop_id_nonorm,specific_person_id_nonorm).detach().cpu().numpy())
                        frame_align_crop_tenor_list.append(frame_align_crop_tenor)
                    id_compare_values_array = np.array(id_compare_values)
                    min_index = np.argmin(id_compare_values_array)
                    min_value = id_compare_values_array[min_index]
                    if min_value < id_thres:
                        swap_result = swap_model(None, frame_align_crop_tenor_list[min_index], id_vetor, None, True)[0]
                        final_img = SimSwap.reverse2wholeimage([frame_align_crop_tenor_list[min_index]], [swap_result], [frame_mat_list[min_index]], crop_size, frame, logoclass,\
                            os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,
                            use_mask= use_mask, norm = spNorm, _kernel_size=_kernel_size, face_part_ids=face_part_ids)

                        ########################################### Two pass #####################################################################################
                        if self_obj.settings.two_pass.get():
                            self_obj.set_status(f"Performing two pass... ({int((frame_index/frame_count)*100)}% completed)")
                            detect_results = detect_model.get(final_img,crop_size)
                            if detect_results is not None:
                                if not os.path.exists(temp_results_dir):
                                        os.mkdir(temp_results_dir)
                                frame_align_crop_list = detect_results[0]
                                frame_mat_list = detect_results[1]
                                swap_result_list = []
                                frame_align_crop_tenor_list = []
                                for frame_align_crop in frame_align_crop_list:
                                    frame_align_crop_tenor = SimSwap._totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
                                    swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                                    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                                    swap_result_list.append(swap_result)
                                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)
                                final_img = SimSwap.reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
                                    os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,
                                    use_mask=use_mask, norm = spNorm, _kernel_size=_kernel_size, face_part_ids=face_part_ids)
                        ########################################### Two pass #####################################################################################

                        try:
                            if self_obj.stop_process:
                                return
                            self_obj.video_player.update_display(image=final_img)
                            self_obj.set_status(f"Processing {frame_index} of {frame_count}... ({int((frame_index/frame_count)*100)}% completed)")
                        except: pass
                    else:
                        if not os.path.exists(temp_results_dir):
                            os.mkdir(temp_results_dir)
                        frame = frame.astype(np.uint8)
                        if not no_simswaplogo:
                            frame = logoclass.apply_frames(frame)
                        cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

                else:
                    if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                    frame = frame.astype(np.uint8)
                    if not no_simswaplogo:
                        frame = logoclass.apply_frames(frame)
                    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                    self_obj.set_status(f"Skipping no face...({int((frame_index/frame_count)*100)}% completed)")
            else:
                break
        video.release()
        self_obj.set_status(f"Merging sequence...")
        path = os.path.join(temp_results_dir,'*.jpg')
        image_filenames = sorted(glob.glob(path))
        clips = ImageSequenceClip(image_filenames,fps = fps)
        if not no_audio:
            clips = clips.set_audio(video_audio_clip)
        clips.write_videofile(save_path,audio_codec='aac')

    def video_swap_multispecific(video_path, target_id_norm_list,source_specific_id_nonorm_list,id_thres, swap_model,
        detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False,
        self_obj=None, _kernel_size=(40, 40), face_part_ids=[1,2]):
        video_forcheck = VideoFileClip(video_path)
        if video_forcheck.audio is None:
            no_audio = True
        else:
            no_audio = False
        del video_forcheck
        if not no_audio:
            video_audio_clip = AudioFileClip(video_path)
        video = cv2.VideoCapture(video_path)
        logoclass = watermark_image('./simswaplogo/simswaplogo.png')
        ret = True
        frame_index = 0
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        if  os.path.exists(temp_results_dir):
                shutil.rmtree(temp_results_dir)
        spNorm =SpecificNorm()
        mse = torch.nn.MSELoss().cuda()
        if use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net =None
        for frame_index in tqdm(range(frame_count)):
            ret, frame = video.read()
            if  ret:
                detect_results = detect_model.get(frame,crop_size)
                if detect_results is not None:
                    if not os.path.exists(temp_results_dir):
                            os.mkdir(temp_results_dir)
                    frame_align_crop_list = detect_results[0]
                    frame_mat_list = detect_results[1]
                    id_compare_values = []
                    frame_align_crop_tenor_list = []
                    for frame_align_crop in frame_align_crop_list:
                        frame_align_crop_tenor = SimSwap._totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
                        frame_align_crop_tenor_arcnorm = spNorm(frame_align_crop_tenor)
                        frame_align_crop_tenor_arcnorm_downsample = F.interpolate(frame_align_crop_tenor_arcnorm, size=(112,112))
                        frame_align_crop_crop_id_nonorm = swap_model.netArc(frame_align_crop_tenor_arcnorm_downsample)
                        id_compare_values.append([])
                        for source_specific_id_nonorm_tmp in source_specific_id_nonorm_list:
                            id_compare_values[-1].append(mse(frame_align_crop_crop_id_nonorm,source_specific_id_nonorm_tmp).detach().cpu().numpy())
                        frame_align_crop_tenor_list.append(frame_align_crop_tenor)
                    id_compare_values_array = np.array(id_compare_values).transpose(1,0)
                    min_indexs = np.argmin(id_compare_values_array,axis=0)
                    min_value = np.min(id_compare_values_array,axis=0)
                    swap_result_list = []
                    swap_result_matrix_list = []
                    swap_result_ori_pic_list = []
                    for tmp_index, min_index in enumerate(min_indexs):
                        if min_value[tmp_index] < id_thres:
                            swap_result = swap_model(None, frame_align_crop_tenor_list[tmp_index], target_id_norm_list[min_index], None, True)[0]
                            swap_result_list.append(swap_result)
                            swap_result_matrix_list.append(frame_mat_list[tmp_index])
                            swap_result_ori_pic_list.append(frame_align_crop_tenor_list[tmp_index])
                        else:
                            pass
                    if len(swap_result_list) !=0:
                        final_img = SimSwap.reverse2wholeimage(swap_result_ori_pic_list,swap_result_list, swap_result_matrix_list, crop_size, frame, logoclass,\
                            os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask=use_mask,
                            norm = spNorm, _kernel_size=_kernel_size, face_part_ids=face_part_ids)

                        ########################################################## Two Pass ####################################################################
                        if self_obj.settings.two_pass.get():
                            self_obj.set_status(f"Performing two pass... ({int((frame_index/frame_count)*100)}% completed)")
                            detect_results = detect_model.get(final_img,crop_size)
                            if detect_results is not None:
                                if not os.path.exists(temp_results_dir):
                                        os.mkdir(temp_results_dir)
                                frame_align_crop_list = detect_results[0]
                                frame_mat_list = detect_results[1]
                                id_compare_values = []
                                frame_align_crop_tenor_list = []
                                for frame_align_crop in frame_align_crop_list:
                                    frame_align_crop_tenor = SimSwap._totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
                                    frame_align_crop_tenor_arcnorm = spNorm(frame_align_crop_tenor)
                                    frame_align_crop_tenor_arcnorm_downsample = F.interpolate(frame_align_crop_tenor_arcnorm, size=(112,112))
                                    frame_align_crop_crop_id_nonorm = swap_model.netArc(frame_align_crop_tenor_arcnorm_downsample)
                                    id_compare_values.append([])
                                    for source_specific_id_nonorm_tmp in source_specific_id_nonorm_list:
                                        id_compare_values[-1].append(mse(frame_align_crop_crop_id_nonorm,source_specific_id_nonorm_tmp).detach().cpu().numpy())
                                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)
                                id_compare_values_array = np.array(id_compare_values).transpose(1,0)
                                min_indexs = np.argmin(id_compare_values_array,axis=0)
                                min_value = np.min(id_compare_values_array,axis=0)
                                swap_result_list = []
                                swap_result_matrix_list = []
                                swap_result_ori_pic_list = []
                                for tmp_index, min_index in enumerate(min_indexs):
                                    if min_value[tmp_index] < id_thres:
                                        swap_result = swap_model(None, frame_align_crop_tenor_list[tmp_index], target_id_norm_list[min_index], None, True)[0]
                                        swap_result_list.append(swap_result)
                                        swap_result_matrix_list.append(frame_mat_list[tmp_index])
                                        swap_result_ori_pic_list.append(frame_align_crop_tenor_list[tmp_index])
                                    else:
                                        pass
                                if len(swap_result_list) !=0:
                                    final_img = SimSwap.reverse2wholeimage(swap_result_ori_pic_list,swap_result_list, swap_result_matrix_list, crop_size, frame, logoclass,\
                                        os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask=use_mask,
                                        norm = spNorm, _kernel_size=_kernel_size, face_part_ids=face_part_ids)
                        ########################################################## Two Pass ####################################################################
                        if self_obj.stop_process:
                            return
                        self_obj.video_player.update_display(image=final_img)
                        self_obj.set_status(f"Processing {frame_index} of {frame_count}... ({int((frame_index/frame_count)*100)}% completed)")
                    else:
                        if not os.path.exists(temp_results_dir):
                            os.mkdir(temp_results_dir)
                        frame = frame.astype(np.uint8)
                        if not no_simswaplogo:
                            frame = logoclass.apply_frames(frame)
                        cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                else:
                    if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                    frame = frame.astype(np.uint8)
                    if not no_simswaplogo:
                        frame = logoclass.apply_frames(frame)
                    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                    self_obj.set_status(f"Skipping no face...({int((frame_index/frame_count)*100)}% completed)")
            else:
                break
        video.release()
        self_obj.set_status(f"Merging sequence...")
        path = os.path.join(temp_results_dir,'*.jpg')
        image_filenames = sorted(glob.glob(path))
        clips = ImageSequenceClip(image_filenames,fps = fps)
        if not no_audio:
            clips = clips.set_audio(video_audio_clip)
        clips.write_videofile(save_path,audio_codec='aac')

    def runSwap(srcImg, dstImg, vidPath, settings, swap_all=False, self_obj=None):
        transformer = transforms.Compose([transforms.ToTensor(),])
        transformer_Arcface = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        opt = TestOptions().parse()
        opt.Arc_path = settings.arc_path.get()
        opt.temp_path = settings.temp_path.get()
        opt.use_mask = settings.use_mask.get()
        opt.output_path = settings.out_path.get()
        opt.video_path = vidPath
        opt.id_thres = 0.03
        opt.no_simswaplogo = not settings.simswap_logo.get()

        start_epoch, epoch_iter = 1, 0
        opt.crop_size = int(settings.crop_size.get())
        crop_size = opt.crop_size
        det_size = int(settings.det_size.get())
        _kernel_size = tuple(map(int, settings.kernel_size.get().split(' ')))
        face_part_ids = list(map(int, settings.face_part_ids.get().split(' ')))
        mode = 'None'
        if settings.ffhq.get():
            mode = 'ffhq'

        torch.nn.Module.dump_patches = True
        if crop_size == 512:
            opt.which_epoch = 550000
            opt.name = '512'
            #mode = 'ffhq'
        #else:
        #    mode = 'None'
        model = create_model(opt)
        model.eval()

        app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        app.prepare(ctx_id= 0, det_thresh=settings.det_thresh.get(), det_size=(det_size,det_size), mode=mode)

        with torch.no_grad():
            img_a_whole = srcImg
            img_a_align_crop, _ = app.get(img_a_whole,crop_size)
            img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB))
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

            img_id = img_id.cuda()

            img_id_downsample = F.interpolate(img_id, size=(112,112))
            latend_id = model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)

            if swap_all:
                img_id_downsample = F.interpolate(img_id, size=(112,112))
                latend_id = model.netArc(img_id_downsample)
                latend_id = F.normalize(latend_id, p=2, dim=1)

                SimSwap.video_swap(
                    opt.video_path,
                    latend_id, model,
                    app,
                    opt.output_path,
                    temp_results_dir=opt.temp_path,\
                    no_simswaplogo=opt.no_simswaplogo,
                    use_mask=opt.use_mask,
                    self_obj=self_obj,
                    _kernel_size=_kernel_size,
                    face_part_ids=face_part_ids)
            else:
                specific_person_whole = dstImg
                specific_person_align_crop, _ = app.get(specific_person_whole,crop_size)
                specific_person_align_crop_pil = Image.fromarray(cv2.cvtColor(specific_person_align_crop[0],cv2.COLOR_BGR2RGB))
                specific_person = transformer_Arcface(specific_person_align_crop_pil)
                specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1], specific_person.shape[2])
                specific_person = specific_person.cuda()
                specific_person_downsample = F.interpolate(specific_person, size=(112,112))
                specific_person_id_nonorm = model.netArc(specific_person_downsample)

                SimSwap.video_target_swap(
                    opt.video_path,
                    latend_id,
                    specific_person_id_nonorm,
                    opt.id_thres,
                    model,
                    app,
                    opt.output_path,
                    temp_results_dir=opt.temp_path,
                    no_simswaplogo=opt.no_simswaplogo,
                    use_mask=opt.use_mask,
                    crop_size=crop_size,
                    self_obj=self_obj,
                    _kernel_size=_kernel_size,
                    face_part_ids=face_part_ids)

    def swap_multi_specific(src_dst_dict, vidPath, settings, self_obj=None):
        transformer = transforms.Compose([transforms.ToTensor(),])
        transformer_Arcface = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        opt = TestOptions().parse()
        opt.Arc_path = settings.arc_path.get()
        opt.temp_path = settings.temp_path.get()
        opt.use_mask = settings.use_mask.get()
        opt.output_path = settings.out_path.get()
        opt.video_path = vidPath
        opt.id_thres = 0.03
        opt.no_simswaplogo = not settings.simswap_logo.get()

        start_epoch, epoch_iter = 1, 0
        opt.crop_size = int(settings.crop_size.get())
        crop_size = opt.crop_size
        det_size = int(settings.det_size.get())
        _kernel_size = tuple(map(int, settings.kernel_size.get().split(' ')))
        face_part_ids = list(map(int, settings.face_part_ids.get().split(' ')))
        mode = 'None'
        if settings.ffhq.get():
            mode = 'ffhq'

        torch.nn.Module.dump_patches = True
        if crop_size == 512:
            opt.which_epoch = 550000
            opt.name = '512'
            #mode = 'ffhq'
        #else:
        #    mode = 'None'
        model = create_model(opt)
        model.eval()

        app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        app.prepare(ctx_id= 0, det_thresh=settings.det_thresh.get(), det_size=(det_size,det_size), mode=mode)

        with torch.no_grad():
            source_specific_id_nonorm_list = []
            target_id_norm_list = []

            for key in src_dst_dict:
                _src = src_dst_dict[key][1]
                _dst = src_dst_dict[key][0]

                specific_person_align_crop, _ = app.get(_src, crop_size)
                specific_person_align_crop_pil = Image.fromarray(cv2.cvtColor(specific_person_align_crop[0],cv2.COLOR_BGR2RGB))
                specific_person = transformer_Arcface(specific_person_align_crop_pil)
                specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1], specific_person.shape[2])
                specific_person = specific_person.cuda()
                specific_person_downsample = F.interpolate(specific_person, size=(112,112))
                specific_person_id_nonorm = model.netArc(specific_person_downsample)
                source_specific_id_nonorm_list.append(specific_person_id_nonorm.clone())

                img_a_align_crop, _ = app.get(_dst,crop_size)
                img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB))
                img_a = transformer_Arcface(img_a_align_crop_pil)
                img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
                img_id = img_id.cuda()
                img_id_downsample = F.interpolate(img_id, size=(112,112))
                latend_id = model.netArc(img_id_downsample)
                latend_id = F.normalize(latend_id, p=2, dim=1)
                target_id_norm_list.append(latend_id.clone())

            assert len(target_id_norm_list) == len(source_specific_id_nonorm_list)

            SimSwap.video_swap_multispecific(
                opt.video_path,
                target_id_norm_list,
                source_specific_id_nonorm_list,
                opt.id_thres,
                model,
                app,
                opt.output_path,
                temp_results_dir=opt.temp_path,
                no_simswaplogo=opt.no_simswaplogo,
                use_mask=opt.use_mask,
                crop_size=crop_size,
                self_obj=self_obj,
                _kernel_size=_kernel_size,
                face_part_ids=face_part_ids)

class Utils:
    def remove_files(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def pil_create_txt_image(size, text, tk_image=True):
        img = Image.new('RGB', size)
        img_draw = ImageDraw.Draw(img)
        w, h = img_draw.textsize(text)
        img_draw.text(((size[0]-w)/2,(size[1]-h)/2), text, fill="white")
        if tk:
            return Utils.pil_to_tkImage(img)
        return img

    def get_error_image(size):
        img = np.ones((size[1], size[0], 3), dtype=np.uint8)
        img[:] = (125, 0, 0)
        return img

    def make_preview_image(img, expected_size=(500, 600), scale_to_fit=False):
        if scale_to_fit:
            base_width = expected_size[0]
            wpercent = (base_width/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((base_width,hsize), Image.ANTIALIAS)

        img.thumbnail((expected_size[0], expected_size[1]))
        delta_width = expected_size[0] - img.size[0]
        delta_height = expected_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        new_img = ImageOps.expand(img, padding)
        return new_img

    def cv2_to_tkImage(img, size=72):
        b,g,r = cv2.split(img)
        img = cv2.merge((r,g,b))
        im = Image.fromarray(img)
        im = im.resize((size,size), resample=Image.LANCZOS)
        return Utils.pil_to_tkImage(im)

    def cv2_to_pil(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def pil_to_tkImage(img):
        return ImageTk.PhotoImage(img)

    class Spinbox(ttk.Entry):
        def __init__(self, master=None, **kw):
            ttk.Entry.__init__(self, master, "ttk::spinbox", **kw)
        def set(self, value):
            self.tk.call(self._w, "set", value)

    class ImageCanvas:
        def __init__(self, window, size, default_text="..."):
            self.size = size
            self.image = Utils.pil_create_txt_image(size, default_text, tk_image=False)
            self.topx, self.topy, self.botx, self.boty = 0, 0, 0, 0
            self.rect_id = None
            self.canvas = tk.Canvas(window, width=self.image.width(), height=self.image.height(),
                    borderwidth=0, highlightthickness=0)
            self.canvas_image = self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)
            self.canvas.img = self.image
            self.canvas.pack(side="left", fill="both", expand="yes")
            self.rect_id = self.canvas.create_rectangle(self.topx, self.topy, self.topx, self.topy,
                                    dash=(2,2), fill='', outline='white')
            self.canvas.bind('<Button-1>', self.get_mouse_posn)
            self.canvas.bind('<B1-Motion>', self.update_sel_rect)

        def get_mouse_posn(self, event):
            self.topx, self.topy = event.x, event.y

        def update_sel_rect(self, event, fixed=False):
            self.botx, self.boty = event.x, event.y
            if fixed:
                maxb = max(self.botx, self.boty)
                maxb -= max(self.topx, self.topy)
                self.botx, self.boty = self.topx + maxb, self.topy + maxb
            self.canvas.coords(self.rect_id, self.topx, self.topy, self.botx, self.boty)

        def update_canvas(self, image):
            self.image = Utils.make_preview_image(image, expected_size=self.size, scale_to_fit=True)
            image = Utils.pil_to_tkImage(self.image)
            self.canvas.itemconfig(self.canvas_image, image=image)
            self.canvas.configure(width=image.width(), height=image.height())
            self.canvas.img = image

        def get_roi(self):
            x1, y1, x2, y2 = self.topx, self.topy, self.botx, self.boty
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            return [x1, y1, x2, y2]

class VideoPlayer:
    def __init__(self, frame, size=(300, 400)):
        self.monitor_frame = ttk.Frame(frame)
        self.monitor_frame.pack(side="top", expand=True, padx=10)
        self.monitor_size = size
        self.pause = True

        # monitor
        self.monitor_img = Utils.ImageCanvas(self.monitor_frame, self.monitor_size, default_text="Simswap GUI")

        self.controller_frame = ttk.Frame(frame)
        self.controller_frame.pack(side="top", expand=True, padx=10)

        # slider
        self.slider = ttk.Scale(self.controller_frame, length=size[0])
        self.slider.pack(side="top", pady=5, expand=True)

        # buttons and controls
        self.sub_controller_frame = ttk.Frame(self.controller_frame)
        self.sub_controller_frame.pack(side="top", anchor=tk.CENTER, pady=5, expand=True)

        self.start_frame_spin = Utils.Spinbox(self.sub_controller_frame, width=5, from_=1)
        self.stop_frame_spin = Utils.Spinbox(self.sub_controller_frame, width=5, from_=1)
        self.forward_btn = ttk.Button(self.sub_controller_frame, text=">>", style='btn.TButton')
        self.bacward_btn = ttk.Button(self.sub_controller_frame, text="<<", style='btn.TButton')
        self.play_btn = ttk.Button(self.sub_controller_frame, text="Play", style='btn.TButton')
        self.pause_btn = ttk.Button(self.sub_controller_frame, text="Pause", style='btn.TButton')
        self.trim_start_btn = ttk.Button(self.sub_controller_frame, text=u"[--", style='btn.TButton')
        self.trim_stop_btn = ttk.Button(self.sub_controller_frame, text="--]", style='btn.TButton')
        self.current_frame_label = ttk.Label(self.sub_controller_frame, text="0001", font='Helvetica 16 bold')

        self.start_frame_spin.grid(column=0, row=0, padx=(0,10))
        self.trim_start_btn.grid(column=1, row=0, padx=2)
        self.bacward_btn.grid(column=2, row=0, padx=2)
        self.pause_btn.grid(column=3, row=0, padx=2)
        self.current_frame_label.grid(column=4, row=0)
        self.play_btn.grid(column=5, row=0, padx=2)
        self.forward_btn.grid(column=6, row=0, padx=2)
        self.trim_stop_btn.grid(column=7, row=0, padx=2)
        self.stop_frame_spin.grid(column=8, row=0, padx=(10,0))

        self.current_frame = tk.IntVar(value=1)
        self.start_frame = tk.IntVar(value=1)
        self.stop_frame = tk.IntVar(value=100)
        self.cap = None

        self.set_monitor_variables()
        self.set_monitor_functions()
        self.reader_available = False

    def interrupt_play(self):
        self.pause = True
        self.play_btn.config(state='normal')

    def set_max_frame(self, value):
        self.start_frame_spin.config(to=value)
        self.stop_frame_spin.config(to=value)

    def set_monitor_variables(self):
        self.slider.config(variable=self.current_frame)
        self.slider.config(from_=self.start_frame.get())
        self.slider.config(to=self.stop_frame.get())
        self.start_frame_spin.config(textvariable=self.start_frame, to=1000)
        self.stop_frame_spin.config(textvariable=self.stop_frame, to=1000)

    def slider_func(self, *args):
        self.slider.config(from_=self.start_frame.get())
        self.slider.config(to=self.stop_frame.get())
        frame = str(max(self.current_frame.get(),1))
        self.current_frame_label.config(text=frame.zfill(4))
        if self.pause:
            self.update_display()

    def trim_start_func(self, *args):
        self.start_frame_spin.set(self.current_frame.get())
        self.slider_func()

    def trim_stop_func(self, *args):
        self.stop_frame_spin.set(self.current_frame.get())
        self.slider_func()

    def set_monitor_functions(self):
        self.slider.config(command=self.slider_func)
        self.trim_start_btn.config(command=self.trim_start_func)
        self.trim_stop_btn.config(command=self.trim_stop_func)
        self.play_btn.config(command=lambda: threading.Thread(target=self.play_video).start())
        self.pause_btn.config(command=self.interrupt_play)
        self.forward_btn.config(command=lambda: self.slider.set(self.current_frame.get() + 1))
        self.bacward_btn.config(command=lambda: self.slider.set(self.current_frame.get() - 1))

    def reset_monitor_controls(self):
        self.start_frame.set(1)
        self.current_frame.set(1)
        self.start_frame_spin.config(from_=1, to=99999)
        self.stop_frame_spin.config(from_=1, to=99999)
        self.slider.config(from_=1, to=99999)
        self.slider.set(1)

    def read_video(self, path="video.mp4"):
        try:
            if self.cap is not None:
                self.cap.release()
            self.input_path = path
            self.cap = cv2.VideoCapture(path)
            self.input_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.reader_available = True
        except Exception as e:
            print(e)
            self.reader_available = False

    def get_video_frame(self, frame=1):
        if frame <= 0 or not self.reader_available:
            return Utils.get_error_image(self.monitor_size)
        self.cap.set(1, frame - 1)
        _, image = self.cap.read()
        return image

    def get_current_frame(self):
        return self.get_video_frame(self.current_frame.get())

    def update_display(self, image=None):
        try:
            if image is None:
                frame = self.current_frame.get()
                image = self.get_video_frame(frame=frame)
            pil_image = Utils.cv2_to_pil(image)
            self.monitor_img.update_canvas(pil_image)
        except: pass

    def open_video(self, path):
        self.reset_monitor_controls()
        self.read_video(path)
        self.start_frame.set(1)
        self.current_frame.set(1)
        self.stop_frame.set(self.total_frames)
        self.set_max_frame(self.total_frames)
        self.update_display()

    def get_video_data(self):
        if not self.reader_available:
            return
        video = {"frame":self.get_current_frame(),
                 "fps": self.input_fps,
                 "start": self.start_frame.get(),
                 "end": self.stop_frame.get()}
        return video

    def play_video(self):
        self.pause = False
        if not self.reader_available:
            return
        self.play_btn.config(state='disabled')
        current = self.current_frame.get()
        self.cap.set(1, current-1)
        _fps = self.input_fps
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True and self.pause == False and current <= self.stop_frame.get():
                self.slider.set(current)
                current += 1
                self.update_display(image=frame)
                if cv2.waitKey(int((1/int(_fps))*1000)) & 0xFF == ord('q'):
                    break
            else:
                break
        self.interrupt_play()

        # _start = self.current_frame.get()
        # _end = self.stop_frame.get()
        # _fps = max(1, self.cap.get(cv2.CAP_PROP_FPS)) * 10
        # print("fps=", _fps)
        # for i in range(_start, _end):
        #     if self.pause:
        #         break
        #     self.slider.set(i)
        #     self.update_display(image=self.get_video_frame(frame=i))
        #     time.sleep(1/_fps)
        # self.interrupt_play()

class MenuBar:
    def __init__(self, root):
        self.menu_bar = tk.Menu(root)
        self.filemenu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.filemenu)

        self.editmenu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edit", menu=self.editmenu)

        helpmenu = tk.Menu(self.menu_bar, tearoff=0)
        url = "https://github.com/harisreedhar/SimSwap-GUI"
        helpmenu.add_command(label="Report", command=lambda: webbrowser.open(url, new=1))
        info = "Unofficial GUI implementaion of simswap"
        helpmenu.add_command(label="About", command=lambda: messagebox.showinfo("About", info))
        self.menu_bar.add_cascade(label="Help", menu=helpmenu)
        root.config(menu=self.menu_bar)

    def set_menubar_functions(self, functions):
        self.filemenu.add_command(label="Import video", command=functions.get("open video"))
        self.filemenu.add_command(label="Import source image", command=functions.get("import source"))
        self.filemenu.add_command(label="Import target image", command=functions.get("import target"))
        self.filemenu.add_separator()
        #self.filemenu.add_command(label="Open output directory", command=functions.get("open out dir"))
        self.filemenu.add_command(label="Clear temp path", command=functions.get("clear temp"))
        self.filemenu.add_command(label="Clear trim path", command=functions.get("clear trim"))
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Quit", command=functions.get("quit"))

        self.editmenu.add_command(label="Reset trim", command=functions.get("reset trim"))
        self.editmenu.add_command(label="Trim & reload", command=functions.get("trim & reload"))

class StatusBar:
    def __init__(self, root):
        self.status = tk.Label(root, text="...", bd=1, relief=tk.RIDGE, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

class SettingsWindow:
    def __init__(self, root, size):
        _width = 50
        _width_2 = 10
        settings_frame = ttk.Frame(root)
        settings_frame.pack(side="top", fill="both", padx=5, pady=5, expand=True)

        self.out_path = tk.StringVar()
        self.out_path.set(OUTPUT_PATH)
        out_path_label = ttk.Label(settings_frame, text="Output video path")
        self.out_path_entry = ttk.Entry(settings_frame, width= _width, textvariable=self.out_path)

        self.arc_path = tk.StringVar()
        self.arc_path.set(ARC_PATH)
        arc_path_label = ttk.Label(settings_frame, text="ArcFace path")
        self.arc_path_entry = ttk.Entry(settings_frame, width= _width, textvariable=self.arc_path)

        self.temp_path = tk.StringVar()
        self.temp_path.set(TEMP_PATH)
        temp_path_label = ttk.Label(settings_frame, text="Temp seq path")
        self.temp_path_entry = ttk.Entry(settings_frame, width= _width, textvariable=self.temp_path)

        self.trim_path = tk.StringVar()
        self.trim_path.set(TRIM_PATH)
        trim_path_label = ttk.Label(settings_frame, text="Trim save path")
        self.trim_path_entry = ttk.Entry(settings_frame, width= _width, textvariable=self.trim_path)

        self.det_thresh = tk.DoubleVar()
        self.det_thresh.set(DET_THRESHOLD)
        det_thresh_label = ttk.Label(settings_frame, text="Detection Threshold")
        self.det_thresh_entry = ttk.Entry(settings_frame, width= _width_2, textvariable=self.det_thresh)

        self.det_size = tk.IntVar()
        self.det_size.set(DET_SIZE)
        det_size_label = ttk.Label(settings_frame, text="Detection Size")
        self.det_size_entry = ttk.Entry(settings_frame, width= _width_2, textvariable=self.det_size)

        self.kernel_size = tk.StringVar()
        self.kernel_size.set(MASK_KERNEL)
        kernel_size_label = ttk.Label(settings_frame, text="Mask Erode Kernel")
        self.kernel_size_entry = ttk.Entry(settings_frame, width= _width_2, textvariable=self.kernel_size)

        self.crop_size = tk.DoubleVar()
        self.crop_size.set(CROP_SIZE)
        crop_size_label = ttk.Label(settings_frame, text="Crop Size")
        self.crop_size_entry = ttk.Entry(settings_frame, width= _width_2, textvariable=self.crop_size)

        self.ffhq = tk.BooleanVar()
        self.ffhq.set(FFHQ)
        self.use_ffhq_check = ttk.Checkbutton(settings_frame, text="ffhq", variable=self.ffhq)

        self.face_part_ids = tk.StringVar()
        self.face_part_ids.set(FACE_PART_IDS)
        self.face_part_ids_entry = ttk.Entry(settings_frame, width= _width, textvariable=self.face_part_ids)

        self.use_mask = tk.BooleanVar()
        self.use_mask.set(USE_MASK)
        self.use_mask_check = ttk.Checkbutton(settings_frame, text="Use Mask", variable=self.use_mask, command=lambda e=self.face_part_ids_entry, v=self.use_mask: self.set_state(e,v))

        self.simswap_logo = tk.BooleanVar()
        self.simswap_logo.set(WATERMARK)
        self.simswap_logo_check = ttk.Checkbutton(settings_frame, text="Watermark", variable=self.simswap_logo)

        self.two_pass = tk.BooleanVar()
        self.two_pass.set(0)
        self.two_pass_check = ttk.Checkbutton(settings_frame, text="Two Pass", variable=self.two_pass)

        out_path_label.grid(sticky="W",column=0, row=1)
        self.out_path_entry.grid(sticky="W",column=1, row=1, padx=(5,0))
        arc_path_label.grid(sticky="W",column=0, row=2)
        self.arc_path_entry.grid(sticky="W",column=1, row=2, padx=(5,0))
        temp_path_label.grid(sticky="W",column=0, row=3)
        self.temp_path_entry.grid(sticky="W",column=1, row=3, padx=(5,0))
        trim_path_label.grid(sticky="W",column=0, row=4)
        self.trim_path_entry.grid(sticky="W",column=1, row=4, padx=(5,0))

        det_thresh_label.grid(sticky="W",column=2, row=1, padx=(5,0))
        self.det_thresh_entry.grid(sticky="W",column=3, row=1, padx=(2,0))

        det_size_label.grid(sticky="W",column=2, row=2, padx=(5,0))
        self.det_size_entry.grid(sticky="W",column=3, row=2, padx=(2,0))

        kernel_size_label.grid(sticky="W",column=2, row=3, padx=(5,0))
        self.kernel_size_entry.grid(sticky="W",column=3, row=3, padx=(2,0))

        crop_size_label.grid(sticky="W",column=2, row=4, padx=(5,0))
        self.crop_size_entry.grid(sticky="W",column=3, row=4, padx=(2,0))

        self.use_mask_check.grid(sticky="W",column=0, row=5)
        self.face_part_ids_entry.grid(sticky="W",column=1, row=5, padx=(5,0))

        #uncomment this to try two pass
        #self.two_pass_check.grid(sticky="W",column=4, row=1, padx=(5,0))
        self.use_ffhq_check.grid(sticky="W",column=4, row=1, padx=(5,0))
        self.simswap_logo_check.grid(sticky="W",column=4, row=2, padx=(5,0))

    def set_state(self, entry, var):
        if var.get() == 0:
            entry.configure(state='disabled')
        else:
            entry.configure(state='normal')

class MainWindow:
    def __init__(self, root, monitor_size=(300,400)):
        self.root = root
        self.monitor_size = monitor_size

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(expand=True)
        self.frame_1 = ttk.Frame(self.main_frame)
        self.frame_1.grid(column=0, row=0, sticky="nw")
        self.frame_2 = ttk.Frame(self.main_frame)
        self.frame_2.grid(column=0, row=1, sticky="nw", pady=5)
        self.frame_3 = ttk.Frame(self.main_frame)
        self.frame_3.grid(column=1, row=0, sticky="nw", pady=5)
        self.frame_4 = ttk.Frame(self.main_frame)
        self.frame_4.grid(column=1, row=1, sticky="nw", pady=5, padx=5)

        self.src_img = None
        self.dst_img = None
        self.video_file_path = None
        self.stop_process = False
        self.multi_specific_data = {}
        self.list_box_counter = 0

        self.menuBar = MenuBar(self.root)
        menu_functions = {
            "open video": self.open_video,
            "open out dir": lambda: subprocess.Popen(["xdg-open", self.settings.temp_path.get()]),
            "clear temp": lambda: Utils.remove_files(self.settings.temp_path.get()),
            "clear trim": lambda: Utils.remove_files(self.settings.trim_path.get()),
            "quit": lambda: self.cancel_process(exit=True),
            "reset trim": lambda: self.video_file_path if self.open_video(file_path=self.video_file_path) else False,
            "trim & reload": self.trim_and_reload,
            "import source": self.select_source_face,
            "import target": self.select_target_face
            }
        self.menuBar.set_menubar_functions(menu_functions)
        self.staus_bar = StatusBar(self.root)
        self.cancel_button = ttk.Button(self.root, text="Cancel", command=self.cancel_process)

        self.video_player = VideoPlayer(self.frame_1, size=self.monitor_size)
        ## drag and drop
        self.video_player.monitor_img.canvas.drop_target_register(DND_FILES)
        self.video_player.monitor_img.canvas.dnd_bind('<<Drop>>', lambda e: self.open_video(file_path=str(e.data).strip('{}'))) # prevent curly braces in path with spaces
        #self.video_player.monitor_img.dnd_bind('<<Drop>>', lambda e: print(e.data))

        self.settings = SettingsWindow(self.frame_2, monitor_size)

        self.frame_2_2 = ttk.Frame(self.frame_3)
        self.frame_2_2.grid(column=0, row=0)
        src_img = Utils.pil_create_txt_image(ICON_SIZE, "Source")
        self.src_btn = ttk.Button(self.frame_2_2, image=src_img, command=lambda: self.select_source_face(from_roi=True))

        ## drag and drop
        self.src_btn.drop_target_register(DND_FILES)
        self.src_btn.dnd_bind('<<Drop>>', lambda e: self.select_source_face(file_path=e.data))
        self.src_btn.grid(column=1, row=0)
        self.src_btn.photo = src_img

        canvas_1_manage = tk.Canvas(self.frame_2_2, width = 12, height = 60)
        canvas_1_manage.grid(row = 0, column = 0)
        canvas_1_manage.create_text(6, 60, text = " Source ", angle = 90, anchor = "w")

        dst_img = Utils.pil_create_txt_image(ICON_SIZE, "Target")
        self.dst_btn = ttk.Button(self.frame_2_2, image=dst_img, command=lambda: self.select_target_face(from_roi=True))
        self.dst_btn.drop_target_register(DND_FILES)
        self.dst_btn.dnd_bind('<<Drop>>', lambda e: self.select_target_face(file_path=e.data))
        self.dst_btn.grid(column=1, row=1)
        self.dst_btn.photo = dst_img

        canvas_2_manage = tk.Canvas(self.frame_2_2, width = 12, height = 60)
        canvas_2_manage.grid(row = 1, column = 0)
        canvas_2_manage.create_text(6, 60, text = " Target ", angle = 90, anchor = "w")
        ttk.Label(self.frame_3, text="Source-Target List").grid(column=0, row=1, pady=(10,0), sticky="w")

        self.list_box = tk.Listbox(self.frame_3, selectmode=tk.EXTENDED, width = 20)
        self.list_box.bind('<<ListboxSelect>>', lambda e: self.list_box_select_preview())
        self.list_box.grid(column=0, row=2, sticky="nsew", padx=(0,10))
        self.list_box_btn_frame = ttk.Frame(self.frame_3)
        self.list_box_btn_frame.grid(column=0 ,row=3, sticky="nsew")

        self.lst_add_btn = ttk.Button(self.list_box_btn_frame, text="+", width=2, command=self.list_box_add_func)
        self.lst_add_btn.grid(row=0, column=1)
        self.lst_remove_btn = ttk.Button(self.list_box_btn_frame, text="-", width=2, command=self.list_box_remove_func)
        self.lst_remove_btn.grid(row=0, column=2)
        self.lst_update_btn = ttk.Button(self.list_box_btn_frame, text="Updt", width=4, command=self.list_box_update_func)
        self.lst_update_btn.grid(row=0, column=3)
        self.lst_clear_btn = ttk.Button(self.list_box_btn_frame, text="Clr", width=3, command=self.list_box_clear_func)
        self.lst_clear_btn.grid(row=0, column=4)

        _btn_width = 17
        self.swap_all_btn = ttk.Button(self.frame_4, text="Swap All", width=_btn_width, command=lambda: threading.Thread(target=self.run_all_swap).start())
        self.swap_target_btn = ttk.Button(self.frame_4, text="Swap Single", width=_btn_width, command=lambda: threading.Thread(target=self.run_target_swap).start())
        self.swap_multi_btn = ttk.Button(self.frame_4, text="Swap Multiple", width=_btn_width, command=lambda: threading.Thread(target=self.run_multi_swap).start())
        self.swap_all_btn.grid(column=0, row=1)
        self.swap_target_btn.grid(column=0, row=2, pady=(2,0))
        self.swap_multi_btn.grid(column=0, row=3, pady=(2,0))

    def prepare_run(func):
        def wrapper(self, *arg, **kwargs):
            self.video_player.interrupt_play()
            self.stop_process = False
            self.cancel_button.pack(fill="both", expand="yes", padx=10, pady=5)
            self.widgets_state(state='disabled')
            res = None
            try:
                res = func(self, *arg, **kwargs)
            except Exception as e:
                print(e)
                self.set_status("Failed. See console window")
            self.cancel_button.pack_forget()
            self.widgets_state(state='normal')
            return res
        return wrapper

    def list_box_select_preview(self, *args):
        if self.list_box.size() > 0:
            try:
                [index] = self.list_box.curselection()
            except ValueError:
                return
            key = self.list_box.get(index)
            data = self.multi_specific_data[key]
            self.src_img = data[0]
            self.dst_img = data[1]
            self.src_dst_btn_img()

    def list_box_add_func(self):
        if self.src_img is None: return
        if self.dst_img is None: return
        val = str(self.list_box_counter).zfill(4)
        key = f"SRC-DST-{val}"
        self.list_box.insert(tk.END, key)
        self.multi_specific_data[key] = [self.src_img, self.dst_img]
        self.list_box_counter += 1

    def list_box_remove_func(self):
        try:
            [index] = self.list_box.curselection()
        except ValueError:
            return
        key = self.list_box.get(index)
        del self.multi_specific_data[key]
        self.list_box.delete(index)
        self.list_box.select_set(max(index, self.list_box.size()-1))

    def list_box_update_func(self):
        if self.src_img is None: return
        if self.dst_img is None: return
        try:
            [index]=self.list_box.curselection()
        except ValueError:
            return
        key = self.list_box.get(index)
        self.multi_specific_data[key] = [self.src_img, self.dst_img]

    def list_box_clear_func(self):
        self.multi_specific_data.clear()
        self.list_box_counter = 0
        self.list_box.delete(0, tk.END)

    def set_status(self, text):
        _text = (text[:100] + '..') if len(text) > 100 else text
        self.staus_bar.status.config(text=_text)

    def select_source_face(self, file_path=None, from_roi=False):
        if from_roi:
            self.src_img = self.get_roi_img()
            self.src_dst_btn_img()
            return
        if file_path is None:
            file_path = tk.filedialog.askopenfilename(initialdir=CWD, title="Select source image")
        if file_path:
            src = cv2.imread(file_path)
            self.src_img = src
            self.src_dst_btn_img()
            self.set_status(f"source face: {file_path}")

    def select_target_face(self, file_path=None, from_roi=False):
        if from_roi:
            self.dst_img = self.get_roi_img()
            self.src_dst_btn_img()
            return
        if file_path is None:
            file_path = tk.filedialog.askopenfilename(initialdir=CWD, title="Select target image")
        if file_path:
            src = cv2.imread(file_path)
            self.dst_img = src
            self.src_dst_btn_img()
            self.set_status(f"source face: {file_path}")

    def get_roi_img(self):
        if not self.video_player.reader_available:
            self.set_status("?")
            return None
        x1, y1, x2, y2 = self.video_player.monitor_img.get_roi()
        cv_image = cv2.cvtColor(np.array(self.video_player.monitor_img.image), cv2.COLOR_RGB2BGR)
        cropped_image = cv_image[y1:y2, x1:x2]
        return cropped_image

    def open_video(self, file_path=None):
        try:
            if file_path == None:
                file_path = tk.filedialog.askopenfilename(initialdir = CWD, title = "Select video")
            if file_path:
                self.video_file_path = file_path
                self.video_player.open_video(path=file_path)
                self.set_status(f"Video: {file_path}")
        except: pass

    def src_dst_btn_img(self):
        if self.src_img is not None:
            src_img = Utils.cv2_to_tkImage(self.src_img, size=ICON_SIZE[0])
            self.src_btn.config(image= src_img)
            self.src_btn.photo = src_img

        if self.dst_img is not None:
            dst_img = Utils.cv2_to_tkImage(self.dst_img, size=ICON_SIZE[0])
            self.dst_btn.config(image= dst_img)
            self.dst_btn.photo = dst_img

    @prepare_run
    def run_target_swap(self):
        if self.dst_img is None:
            self.set_status("Target face not selected")
            return
        if self.src_img is None:
            self.set_status("Source face not selected")
            return
        self.set_status("Starting single target face swap...")
        SimSwap.runSwap(self.src_img, self.dst_img, self.video_file_path, self.settings, self_obj=self)
        self.set_status("Done.")

    @prepare_run
    def run_multi_swap(self):
        if len(self.multi_specific_data) == 0:
            self.set_status("List is empty")
            return
        self.set_status("Starting multi target face swap...")
        SimSwap.swap_multi_specific(self.multi_specific_data, self.video_file_path, self.settings, self_obj=self)
        self.set_status("Done.")

    @prepare_run
    def run_all_swap(self):
        if self.src_img is None:
            self.set_status("Source face not selected")
            return
        self.set_status("Starting all face swap...")
        SimSwap.runSwap(self.src_img, self.dst_img, self.video_file_path, self.settings, swap_all=True, self_obj=self)
        self.set_status("Done.")

    @prepare_run
    def trim_and_reload(self):
        if not self.video_player.reader_available:
            return
        trim_path = str(self.settings.trim_path.get())
        if not os.path.exists(trim_path):
            os.mkdir(trim_path)
        data = self.video_player.get_video_data()
        start = data.get('start')
        end = data.get('end')
        fps = data.get('fps')
        _start = float(start) / float(fps)
        _end = float(end) / float(fps)
        file_name = str(uuid.uuid4())
        file_path = os.path.join(trim_path, f'{file_name}.mp4')
        ffmpeg_extract_subclip(self.video_file_path, _start, _end, targetname=file_path)
        self.open_video(file_path=file_path)
        self.set_status(f'Trimmed video loaded from: {file_path}')

    def cancel_process(self, exit=False):
        self.set_status("Cancelling...")
        self.stop_process = True
        self.cancel_button.pack_forget()
        if exit: self.root.quit()

    def widgets_state(self, state='disable'):
        def set_child_state(widget):
            for child in widget.winfo_children():
                wtype = child.winfo_class()
                if wtype not in ('TFrame','TLabelframe', 'Frame', 'Labelframe', 'Menu', 'TLabel'):
                    child.configure(state=state)
                else:
                    set_child_state(child)
        set_child_state(self.frame_1)
        set_child_state(self.frame_2)
        set_child_state(self.frame_3)
        set_child_state(self.frame_4)

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    root.style = ttk.Style()
    style = ttk.Style(root)
    theme = THEME_STYLE
    if theme in ttk.Style().theme_names():
        style.theme_use(theme)
    style.configure('btn.TButton')
    root.title("Simswap")
    root.resizable(width=False, height=False)
    app = MainWindow(root, monitor_size=WINDOW_SIZE)
    root.mainloop()
