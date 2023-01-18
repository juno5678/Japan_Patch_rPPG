import math
import random
from face_detection import FaceDetection
from scipy import signal
import cv2
from sklearn.decomposition import FastICA
import sys
from video_realsense_file import Video
import process as ps
import numpy as np


class FPS_HRE(object):
    def __init__(self, dataPath):
        self.sampling_rate = 30  # frame rate
        self.order = 10  # order of butterworth filter
        self.length = 10  # seconds length
        self.buffer_size = self.sampling_rate * self.length
        self.fps = 0.  # represent the performance of the computer, doesn't have any meaning besides that
        self.K = 500  # select K times patches
        self.Tr = 2  # heart rate reliable threshold
        self.Tv = 10  # heart rate vote count threshold
        self.dataPath = dataPath
        self.input_rs_video = Video()
        self.processing = True
        self.fd = FaceDetection()
        self.g_face_buffer = []
        self.g_face_rect_buffer = []
        self.nir_face_buffer = []
        self.nir_face_rect_buffer = []
        self.vote_box_pair_G = np.zeros(199)  # 42 bpm ~ 240 bpm vote box for pair G patches
        self.vote_box_pair_NIR = np.zeros(199)  # 42 bpm ~ 240 bpm vote box for pair NIR patches
        self.vote_box_G_NIR = np.zeros(199)  # 42 bpm ~ 240 bpm vote box for G & NIR patches
        self.final_vote_box = np.zeros(199)  # 42 bpm ~ 240 bpm vote box for final bpm

    def run(self):
        data_input = self.input_rs_video
        data_input.dataPath = self.dataPath
        data_input.start()

        # store face ROI
        for i in range(self.buffer_size):
            rgb_frame, nir_frame = data_input.get_frame()
            rgb_face, rgb_face_rect = self.fd.face_detect_rgb(rgb_frame)
            g_face = rgb_face[:, :, 0]
            self.g_face_buffer.append(g_face)
            self.g_face_rect_buffer.append(rgb_face_rect)
            if nir_frame is not None:
                nir_face, nir_face_rect = self.fd.face_detect_gray(nir_frame)
                self.nir_face_buffer.append(nir_face)
                self.nir_face_rect_buffer.append(nir_face_rect)

        for k in range(self.K):
            # select pair G patches
            g_patch1_rect, g_patch2_rect = ps.face_patch_selection(self.g_face_rect_buffer[i])
            # select pair NIR patches
            nir_patch1_rect, nir_patch2_rect = ps.face_patch_selection(self.nir_face_rect_buffer[i])
            # select G & NIR patches
            g_patch_rect, nir_patch_rect = ps.face_patch_selection(self.g_face_rect_buffer[i],
                                                                   self.nir_face_rect_buffer[i])
            for i in range(self.buffer_size):
                # pair G patches
                g_patch_mean = ps.cal_mean(g_patch1_rect, g_patch2_rect, self.g_face_buffer[i])
                bpm_pair_g = self.signal_process(g_patch_mean)
                if bpm_pair_g is not None:
                    self.vote_box_pair_G[bpm_pair_g - 42] += 1
                # pair NIR patches
                if nir_frame is not None:
                    nir_patch_mean = ps.cal_mean(nir_patch1_rect, nir_patch2_rect, self.nir_face_buffer[i])
                    bpm_pair_nir = self.signal_process(nir_patch_mean)
                    if bpm_pair_nir is not None:
                        self.vote_box_pair_NIR[bpm_pair_nir - 42] += 1
                    # G & NIR patches
                    g_nir_patch_mean = ps.cal_mean(g_patch_rect, nir_patch_rect, self.g_face_buffer[i],
                                                   self.nir_face_buffer[i])
                    bpm_g_nir = self.signal_process(g_nir_patch_mean)
                    if bpm_g_nir is not None:
                        self.vote_box_G_NIR[bpm_g_nir - 42] += 1
        self.final_vote_box = self.combine_vote_box(self.vote_box_pair_G, self.vote_box_pair_NIR, self.vote_box_G_NIR)
        final_bpm = self.find_final_bpm(self.final_vote_box)
        print(final_bpm)

    def find_final_bpm(self, final_vote_box):
        x = np.arange(42, 240, 1)
        coeff = np.polyfit(x, final_vote_box, 2)
        print(coeff)
        a = coeff[0]
        b = coeff[1]

        return -b / (2 * a)

    def combine_vote_box(self, vote_box_pair_G, vote_box_pair_NIR=None, vote_box_G_NIR=None):
        length = vote_box_pair_G.shape[0]
        final_vote_box = np.zeros(length)
        for i in range(length):
            if vote_box_pair_G[i] > self.Tv:
                final_vote_box[i] += vote_box_pair_G[i]
            if vote_box_pair_NIR[i] > self.Tv:
                final_vote_box[i] += vote_box_pair_NIR[i]
            if vote_box_pair_G[i] > self.Tv:
                final_vote_box[i] += vote_box_G_NIR[i]
        return final_vote_box

    def signal_process(self, patches_mean):
        inputs = patches_mean.copy()
        MAFed = ps.moving_average_filter(inputs, 3)
        ICAed = ps.ICA(MAFed, num=2)
        PG = ps.select_pg_signal(inputs, ICAed)
        detrended = ps.detrend(PG)
        normalized = ps.normalize(detrended)
        sec_MAFed = ps.moving_average_filter(normalized, 3)
        ppg = ps.butter_bandpass_filter(sec_MAFed, 0.67, 2.68, self.sampling_rate, self.order)
        frequency, PSD = signal.welch(ppg, self.sampling_rate)
        bpm = ps.check_and_get_bpm(PSD, self.buffer_size, self.sampling_rate, self.tr)

        return bpm


if __name__ == '__main__':

    input_realsense = Video()
    dataPath = sys.argv[1]
    HR_Estimator = FPS_HRE(dataPath)
    HR_Estimator.run()
