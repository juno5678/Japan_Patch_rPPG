import numpy as np
import time
import scipy
import math
import random
from face_detection import FaceDetection
from scipy import signal
from scipy.sparse import spdiags
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
"""
def get_patches_signal(K, frame1, frame1_key_landmark, frame2 = None, frame2_key_landmark = None):
    if frame2 is not None:
        # get patches mean
        patches_weights = [None]*2
        patches_rect = [None]*2
        patches_mean = [None]*K
        for k in range(K):
            for i in range(2):
                patches_weights[i] = ps.get_patch_weights()
                patches_rect[i] = ps.get_patch(g_patches_weights[i][k], g_key_landmark)
            g_patches_mean[k] = ps.cal_mean(g_patches_rect[0], g_patches_rect[1], g_frame)
"""
# get left eye, right eye and mouse's landmark
def get_key_landmark(frame, t, fd):
    # detect face
    if t == 0:
        face_rects = fd.face_detect(frame)
    else:
        face_rects = fd.face_track(frame)

    # detect 68 landmark
    landmark = fd.detect_landmark(frame, face_rects)
    # get key landmark
    return fd.get_key_landmark(landmark)


# get the random weights for the height and width of the patches
def get_patch_weights():
    return [random.random(), random.random()]


# get the upper left point & size of the patch
def get_patch(weights, key_landmark):
    leftEye = key_landmark[0]
    rightEye = key_landmark[1]
    mouse = key_landmark[2]
    top_y = min(leftEye.y, rightEye.y)
    bottom_y = mouse.y
    left_x = leftEye.x
    right_x = rightEye.x
    width = right_x - left_x
    height = bottom_y - top_y
    patch_width = round(0.3*width)
    patch_height = round(0.3*height)
    patch_x = round(weights[0] * (0.7 * width) + left_x)
    patch_y = round(weights[1] * (0.7 * height) + top_y)
    return [patch_x, patch_y, patch_width, patch_height]

def detrend(inputs, lamb=1000):
    filtered_signal = np.empty((2, 1))

    x = 0

    for i in np.arange(inputs.shape[0]):
        input_signal = inputs[i]
        signal_length = input_signal.shape[0]
        # observation matrix
        H = np.identity(signal_length)

        # second-order difference matrix
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diagonal_data = np.array([ones, minus_twos, ones])
        diagonal_index = np.array([0, 1, 2])
        D = spdiags(diagonal_data, diagonal_index, (signal_length - 2), signal_length).toarray()
        filtered_signal[i] = np.dot((H - np.linalg.inv(H + (lamb ** 2) * np.dot(D.T, D))), input_signal)

    return filtered_signal


def ICA(observations, num):
    transformer = FastICA(n_components=num, random_state=0, whiten='unit-variance')
    x_tansformed = transformer.fit_transform(observations)
    return x_tansformed


def butter_bandpass_filter(input_signal, lowcut, highcut, fs, order):
    filtered_signal = np.empty((2, 1))

    for i in np.arange(input_signal.shape[0]):
        noisy_signal = input_signal[i]
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        filtered_signal[i] = signal.sosfilt(sos, noisy_signal)

    return filtered_signal


# not align vision, second patch not same with first patch in different channel
def face_patch_selection(frame1_rect, frame2_rect=None):
    patch1_x = random.randint(0, frame1_rect[0] - frame1_rect[2] * 0.3)
    patch1_y = random.randint(0, frame1_rect[1] - frame2_rect[3] * 0.3)

    if frame2_rect is None:
        patch2_x = random.randint(0, frame1_rect[0] - frame1_rect[2] * 0.3)
        patch2_y = random.randint(0, frame1_rect[1] - frame1_rect[3] * 0.3)
    else:
        patch2_x = random.randint(0, frame2_rect[0] - frame2_rect[2] * 0.3)
        patch2_y = random.randint(0, frame2_rect[1] - frame2_rect[3] * 0.3)

    # get patch
    patch1_rect = (patch1_x, patch1_y, frame1_rect[2], frame1_rect[3])
    patch2_rect = (patch2_x, patch2_y, frame2_rect[2], frame2_rect[3])

    return patch1_rect, patch2_rect


def cal_mean(patch1_rect, patch2_rect, frame1, frame2=None):
    patch1 = frame1[patch1_rect[1]:patch1_rect[1] + patch1_rect[3], patch1_rect[0]:patch1_rect[0] + patch1_rect[2]]
    if frame2 is None:
        patch2 = frame1[patch2_rect[1]:patch2_rect[1] + patch2_rect[3], patch2_rect[0]:patch2_rect[0] + patch2_rect[2]]
    else:
        patch2 = frame2[patch2_rect[1]:patch2_rect[1] + patch2_rect[3], patch2_rect[0]:patch2_rect[0] + patch2_rect[2]]

    patch1_mean = np.mean(patch1)
    patch2_mean = np.mean(patch2)

    return np.stack((patch1_mean, patch2_mean), axis=0).reshape((2, 1))


def moving_average_filter(inputs, window_size):
    moving_average = np.empty((2, 1))
    x = 0

    for i in np.arange(inputs.shape[0]):
        signal_input = inputs[i]
        while x < len(signal_input) - window_size + 1:
            window = signal_input[x:x + window_size]
            window_average = round(sum(window) / window_size, 2)
            moving_average[i].append(window_average)
            x += 1

    return moving_average


def normalize(inputs):
    normalized = np.empty((2, 1))

    for i in np.arange(inputs.shape[0]):
        signal_input = inputs[i]
        normalized[i] = (inputs[i] - np.mean(inputs[i])) / np.std(inputs[i])

    return normalized


def check_and_get_bpm(psd, frequency, buffer_size, sampling_rate, tr=2):
    frequency = scipy.fft.rfftfreq(n=5 * buffer_size, d=(1 / sampling_rate))
    frequency *= 60
    fst_max_idx = np.argmax(psd)
    wo_max_PSD_pair_PG = psd
    wo_max_PSD_pair_PG[fst_max_idx] = np.min(psd)
    sec_max_idx = np.argmax(wo_max_PSD_pair_PG)
    v1 = frequency[fst_max_idx]
    v2 = frequency[fst_max_idx]

    # if len(self.bpms) == 0:
    #    S = np.sum(self.PSD[idx-5:idx+6])
    #    self.PSD[idx-5:idx+6] = 0
    #    print('SNR: ' + str(S/np.sum(self.PSD)))

    if v1 / v2 > tr:
        return round(v1)
    else:
        return None


def select_pg_signal(raw_signal, ica_signal):
    max_dis = 0
    max_dis_index = 0
    mean = []
    for i in np.arange(raw_signal.shape[0]):
        mean[i] = np.mean(raw_signal[i])
    avg_mean = np.mean(mean)

    for i in np.arange(ica_signal.shape[0]):
        temp = math.sqrt(pow((np.mean(ica_signal[i]) - avg_mean), 2))
        if temp > max_dis:
            max_dis = temp
            max_dis_index = i

    return ica_signal[max_dis_index]



