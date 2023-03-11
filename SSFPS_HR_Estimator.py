import time

import matplotlib.pyplot as plt
import sklearn.exceptions

from face_detection import FaceDetection
from scipy import signal
import cv2
import sys
from video_realsense_file import Video
from video_rgb_only import Video_RGB
import process as ps
import numpy as np
import os
import warnings

ConvergenceWarning_count = 0
def warning_counter(message, category, filename, lineno, file=None, line=None):
    global ConvergenceWarning_count
    if category == sklearn.exceptions.ConvergenceWarning:
        ConvergenceWarning_count += 1
    s = warnings.formatwarning(message, category, filename, lineno, line)
    #print(category)
    #print(s)


class FPS_HRE(object):
    def __init__(self, dataPath):
        self.sampling_rate = 30  # frame rate
        self.order = 10  # order of butterworth filter
        self.length = 10  # seconds length
        self.buffer_size = self.sampling_rate * self.length
        self.fps = 0.  # represent the performance of the computer, doesn't have any meaning besides that
        self.K = 500  # select K times patches
        self.Tr = 1.5  # heart rate reliable threshold
        self.Tv = 10  # heart rate vote count threshold
        self.dataPath = dataPath
        self.input_rs_video = Video()
        self.fd = FaceDetection()
        self.vote_box_pair_G = np.zeros(199)  # 42 bpm ~ 240 bpm vote box for pair G patches
        self.vote_box_pair_NIR = np.zeros(199)  # 42 bpm ~ 240 bpm vote box for pair NIR patches
        self.vote_box_G_NIR = np.zeros(199)  # 42 bpm ~ 240 bpm vote box for G & NIR patches
        self.final_vote_box = np.zeros(199)  # 42 bpm ~ 240 bpm vote box for final bpm
        self.mode = 1  # 0 : rgb, 1 : rgb+NIR
        self.mode_init = False
        self.patch_num = 2
        self.ma_window_size = 10
        self.ica_parameter = {"whiten": "arbitrary-variance", "algorithm": "parallel", "fun": "cube", "whiten_solver": "eigh"}

    def set_data_length(self, length):
        self.length = length
        self.buffer_size = self.length * self.sampling_rate

    def set_dataPath(self, dataPath):
        self.dataPath = dataPath

    def set_ma_window_size(self, ma_window_size):
        self.ma_window_size = ma_window_size

    def set_ica_parameter(self, ica_parameter):
        self.ica_parameter = ica_parameter

    def reset(self):
        self.mode_init = False
        self.vote_box_pair_G = np.zeros(199)  # 42 bpm ~ 240 bpm vote box for pair G patches
        self.vote_box_pair_NIR = np.zeros(199)  # 42 bpm ~ 240 bpm vote box for pair NIR patches
        self.vote_box_G_NIR = np.zeros(199)  # 42 bpm ~ 240 bpm vote box for G & NIR patches
        self.final_vote_box = np.zeros(199)  # 42 bpm ~ 240 bpm vote box for final bpm
        global ConvergenceWarning_count
        ConvergenceWarning_count = 0

    def run(self):

        # check file is exist
        if os.path.isfile(self.dataPath):
            print('file exist')

            # check file extension
            root, extension = os.path.splitext(self.dataPath)
            if extension == '.avi':
                data_input = Video_RGB()
            elif extension == '.bag':
                data_input = Video()
            else:
                print('wrong file extension . Need input .avi or .bag file. ')
                return -1

            # set input data
            data_input.dataPath = self.dataPath
            data_input.start()
        else:
            print('file not exist')
            return -1

        start = time.time()

        # not use first two frame
        #for n in range(2):
        #    rgb_frame, nir_frame = data_input.get_frame()

        # collect data
        for t in range(self.buffer_size):

            rgb_frame, nir_frame = data_input.get_frame()
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            g_frame = rgb_frame[:, :, 1]

            # only RGB mode
            if self.mode == 0:
                # initial mode
                if not self.mode_init:
                    # set patches weights
                    pair_g_patches_weights = [[0]*self.patch_num for i in range(self.K)]
                    pair_g_patches_rect = [None]*self.patch_num
                    pair_g_patches_mean = [[[None]*self.buffer_size for i in range(self.patch_num)] for j in range(self.K)]
                    for k in range(self.K):
                        for i in range(2):
                            pair_g_patches_weights[k][i] = ps.get_patch_weights()
                    g_fd = FaceDetection()
                    self.mode_init = True

                # get key landmark
                g_key_landmark = ps.get_key_landmark(rgb_frame, t, g_fd)

                # get patches mean
                for k in range(self.K):
                    for i in range(self.patch_num):
                        pair_g_patches_rect[i] = ps.get_patch(pair_g_patches_weights[k][i], g_key_landmark)
                    means = ps.cal_mean(pair_g_patches_rect[0], pair_g_patches_rect[1], g_frame)
                    for i in range(self.patch_num):
                        pair_g_patches_mean[k][i][t] = means[i]

            # RGB + NIR mode
            elif self.mode == 1:
                # check NIR information
                if nir_frame is None:
                    print('Not found NIR frame!')
                    return -1

                # initial mode
                if not self.mode_init:
                    # set patches weights
                    pair_g_patches_weights = [[0]*2 for i in range(self.K)]
                    pair_nir_patches_weights = [[0]*2 for i in range(self.K)]
                    pair_g_patches_rect = [None]*2
                    pair_g_patches_mean = [[[None]*self.buffer_size for i in range(self.patch_num)] for j in range(self.K)]
                    pair_nir_patches_rect = [None]*2
                    pair_nir_patches_mean = [[[None]*self.buffer_size for i in range(self.patch_num)] for j in range(self.K)]
                    g_nir_patches_rect = [None]*2
                    g_nir_patches_mean = [[[None]*self.buffer_size for i in range(self.patch_num)] for j in range(self.K)]
                    g_nir_patches_weights = [0] * self.K
                    for k in range(self.K):
                        for i in range(2):
                            pair_g_patches_weights[k][i] = ps.get_patch_weights()
                            pair_nir_patches_weights[k][i] = ps.get_patch_weights()
                        g_nir_patches_weights[k] = ps.get_patch_weights()
                    g_fd = FaceDetection()
                    nir_fd = FaceDetection()
                    self.mode_init = True

                # get key landmark
                g_key_landmark = ps.get_key_landmark(rgb_frame, t, g_fd)
                nir_key_landmark = ps.get_key_landmark(nir_frame, t, nir_fd)

                # get pair g patches mean
                for k in range(self.K):
                    for i in range(self.patch_num):
                        pair_g_patches_rect[i] = ps.get_patch(pair_g_patches_weights[k][i], g_key_landmark)
                    means = ps.cal_mean(pair_g_patches_rect[0], pair_g_patches_rect[1], g_frame)
                    for i in range(self.patch_num):
                        pair_g_patches_mean[k][i][t] = means[i]

                # get pair nir patches mean
                for k in range(self.K):
                    for i in range(self.patch_num):
                        pair_nir_patches_rect[i] = ps.get_patch(pair_nir_patches_weights[k][i], nir_key_landmark)
                    means = ps.cal_mean(pair_nir_patches_rect[0], pair_nir_patches_rect[1], nir_frame)
                    for i in range(self.patch_num):
                        pair_nir_patches_mean[k][i][t] = means[i]

                # get g and nir patches mean
                for k in range(self.K):
                    g_nir_patches_rect[0] = ps.get_patch(g_nir_patches_weights[k], g_key_landmark)
                    g_nir_patches_rect[1] = ps.get_patch(g_nir_patches_weights[k], nir_key_landmark)
                    means = ps.cal_mean(g_nir_patches_rect[0], g_nir_patches_rect[1], g_frame, nir_frame)
                    for i in range(self.patch_num):
                        g_nir_patches_mean[k][i][t] = means[i]

        # estimate heart rate
        if self.mode == 0:
            pair_g_success_count = 0
            for k in range(self.K):
                # signal process and estimate heart rate
                pair_g_bpm = self.signal_process(np.array(pair_g_patches_mean[k]))
                #print('bpm', pair_g_bpm)
                if pair_g_bpm is not None:
                    self.vote_box_pair_G[pair_g_bpm - 42] += 1
                    pair_g_success_count += 1

                #print('vote box pair g', self.vote_box_pair_G)
            #final_bpm = self.find_final_bpm(self.vote_box_pair_G)
            self.final_vote_box = self.combine_vote_box(self.vote_box_pair_G)
            final_bpm = self.find_final_bpm(self.final_vote_box)
            print('final bpm : ', final_bpm)
            print('success count pair G : %d ' % (pair_g_success_count))
        elif self.mode == 1:
            pair_g_success_count = 0
            pair_nir_success_count = 0
            g_nir_success_count = 0
            for k in range(self.K):
                # signal process and estimate heart rate
                pair_g_bpm = self.signal_process(np.array(pair_g_patches_mean[k]))
                pair_nir_bpm = self.signal_process(np.array(pair_nir_patches_mean[k]))
                g_nir_bpm = self.signal_process(np.array(g_nir_patches_mean[k]))
                if pair_g_bpm is not None:
                    self.vote_box_pair_G[pair_g_bpm - 42] += 1
                    pair_g_success_count += 1
                if pair_nir_bpm is not None:
                    self.vote_box_pair_NIR[pair_nir_bpm - 42] += 1
                    pair_nir_success_count += 1
                if g_nir_bpm is not None:
                    self.vote_box_G_NIR[g_nir_bpm - 42] += 1
                    g_nir_success_count += 1

            self.final_vote_box = self.combine_vote_box(self.vote_box_pair_G, self.vote_box_pair_NIR, self.vote_box_G_NIR)
            final_bpm = self.find_final_bpm(self.final_vote_box)
            print('final bpm : ', final_bpm)
            print('success count pair G : %d , pair nir : %d , g and nir : %d' % (pair_g_success_count, pair_nir_success_count, g_nir_success_count))
            print('total success count : %d' % (pair_g_success_count + pair_nir_success_count + g_nir_success_count))
        end = time.time()
        data_input.stop()

        final_fps = self.buffer_size/(end-start)
        #print('second : ', end-start)
        print('fps : ', final_fps)
        print("Number of Convergence warnings:", ConvergenceWarning_count)

        return final_bpm, final_fps
            #cv2.imshow("rgb", rgb_frame)
            #if nir_frame is not None:
            #    cv2.imshow("nir", nir_frame)
            #if cv2.waitKey(1) == 27:
            #    break
        #rgb_face, rgb_face_rect = self.fd.face_detect_rgb(rgb_frame)
        #g_face = rgb_face[:, :, 0]

    def find_final_bpm(self, final_vote_box):
        idx = np.where(final_vote_box > 0)
        if len(idx[0]) != 0 :
            x = np.arange(42, 241, 1)
            max_count = np.max(final_vote_box)
            bar_x = x[idx]
            bar_y = final_vote_box[idx]
            max_idx = np.argmax(final_vote_box)
            pt_idx = np.array([max_idx-1, max_idx, max_idx+1])
            pt_x = pt_idx+42
            if max_idx == final_vote_box.shape[0]-1:
                pt_y = [final_vote_box[pt_idx[0]], final_vote_box[pt_idx[1]], 0]
            elif max_idx == 0:
                pt_y = [0, final_vote_box[pt_idx[1]], final_vote_box[pt_idx[2]]]
            else:
                pt_y = final_vote_box[pt_idx]
            #print('length of x : ', len(x))
            #print('length of final vote box :', len(final_vote_box))
            #print('final vote box :', final_vote_box)
            coeff = np.polyfit(pt_x, pt_y, 2)
            print(coeff)
            a = coeff[0]
            b = coeff[1]
            c = coeff[2]
            line_x = np.arange(pt_x[0], pt_x[-1]+0.1, 0.1)
            line_y = a*np.square(line_x)+b*line_x+c

            ax = plt.gca()
            plt.plot(line_x, line_y, c='green')
            plt.title('vote box', fontsize=24)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.xlabel('HR (BPM)')
            plt.ylabel('count')
            x_major_locator = plt.MultipleLocator(1)
            y_major_locator = plt.MultipleLocator(50)
            ax.xaxis.set_major_locator(x_major_locator)
            ax.yaxis.set_major_locator(y_major_locator)
            plt.xlim(np.min(bar_x)-1.5, np.max(bar_x)+1.5)
            plt.ylim(0, max_count+50)
            #plt.figure()
            #plt.semilogy(pt_x, pt_y)
            plt.bar(bar_x, bar_y, width=0.8)
            plt.scatter(pt_x, pt_y, c='m', s=50, marker='D')
            file = os.path.split(self.dataPath)
            filename = os.path.splitext(file[1])
            save_path = './result/'+str(filename[0])+'_'+str(self.mode)+'.jpg'
            plt.savefig(save_path)
            print('plt save path', save_path)
            #plt.show()
            plt.close()
            return -b / (2 * a)
        else:
            return 0

    def combine_vote_box(self, vote_box_pair_G, vote_box_pair_NIR=None, vote_box_G_NIR=None):
        length = vote_box_pair_G.shape[0]
        final_vote_box = np.zeros(length)
        #print('pair g vote box : ', vote_box_pair_G)
        #print('pair nir vote box : ', vote_box_pair_NIR)
        #print('g nir vote box : ', vote_box_G_NIR)
        for i in range(length):
            if vote_box_pair_G[i] > self.Tv:
                final_vote_box[i] += vote_box_pair_G[i]
            if vote_box_pair_NIR is not None:
                if vote_box_pair_NIR[i] > self.Tv:
                    final_vote_box[i] += vote_box_pair_NIR[i]
                if vote_box_pair_G[i] > self.Tv:
                    final_vote_box[i] += vote_box_G_NIR[i]
        #print('final vote box : ', final_vote_box)
        return final_vote_box

    def signal_process(self, patches_mean):
        inputs = patches_mean.copy()
        MAFed = [[] for i in range(inputs.shape[0])]
        #print('shape of inputs', inputs.shape)
        for i in np.arange(inputs.shape[0]):
            MAFed[i] = ps.moving_average_filter(inputs[i], self.ma_window_size)
        MAFed = np.array(MAFed)
        #print('shape of MADed', MAFed.shape)
        ICAed = ps.ICA(MAFed, num=2, max_iter_n=200, ica_parameter=self.ica_parameter)
        #print('shape of ICAed', ICAed.shape)
        PG = ps.select_pg_signal(inputs, ICAed)
        #print('shape of PG', PG.shape)
        detrended = ps.detrend(PG)
        #print('shape of detrended', detrended.shape)
        #normalized = ps.normalize(detrended)
        #print('shape of normalized', normalized.shape)
        sec_MAFed = ps.moving_average_filter(detrended, int(self.ma_window_size/2))
        #print('shape of sec_MAFed', sec_MAFed.shape)
        ppg = ps.butter_bandpass_filter(sec_MAFed, 0.7, 2.4, self.sampling_rate, self.order)
        #print('shape of ppg', ppg.shape)
        input_length = self.buffer_size-int(self.ma_window_size*1.5)+2
        if input_length < 256:
            nperseg_length = input_length
        else:
            nperseg_length = 256
        frequency, PSD = signal.welch(ppg, self.sampling_rate, nperseg=nperseg_length)
        #plt.figure()
        #plt.semilogy(frequency, np.sqrt(PSD))
        #plt.xlabel('frequency [Hz]')
        #plt.ylabel('PSD')
        #plt.show()
        #print('shape of psd', PSD.shape)
        #print('shape of frequency', frequency.shape)
        bpm = ps.check_and_get_bpm(PSD, frequency, self.buffer_size, self.sampling_rate, self.Tr)



        return bpm


if __name__ == '__main__':

    warnings.simplefilter("always")
    warnings.showwarning = warning_counter
    #input_realsense = Video()
    dataPath = sys.argv[1]
    HR_Estimator = FPS_HRE(dataPath)
    bpm, fps = HR_Estimator.run()
