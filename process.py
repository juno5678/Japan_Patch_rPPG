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


# get left eye, right eye and mouse's landmark
def get_key_landmark(frame, t, fd):
    # detect face
    #print('t', t)
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
    #input_signal = inputs[i]
    #print('shape of inputs signal :', input_signal.shape)
    signal_length = inputs.shape[0]
    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diagonal_data = np.array([ones, minus_twos, ones])
    diagonal_index = np.array([0, 1, 2])
    D = spdiags(diagonal_data, diagonal_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (lamb ** 2) * np.dot(D.T, D))), inputs)

    return filtered_signal


def ICA(observations, num, max_iter_n, ica_parameter):
    transformer = FastICA(n_components=num, random_state=0, whiten=ica_parameter['whiten'], max_iter=max_iter_n,
                          fun=ica_parameter["fun"], whiten_solver=ica_parameter["whiten_solver"],
                          algorithm=ica_parameter["algorithm"])
    x_tansformed = transformer.fit_transform(observations.T)
    x_tansformed = x_tansformed.T
    return x_tansformed


def butter_bandpass_filter(input_signal, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    filtered_signal = signal.sosfilt(sos, input_signal)

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

    return [patch1_mean, patch2_mean]


def moving_average_filter(inputs, window_size):
    moving_average = []
    x = 0
    while x < (inputs.shape[0] - window_size + 1):
        window = inputs[x:x + window_size]
        window_average = round(sum(window) / window_size, 2)
        moving_average.append(window_average)
        x += 1
    return np.array(moving_average)


def normalize(inputs):
    normalized = (inputs - np.mean(inputs)) / np.std(inputs)
    return normalized


def check_and_get_bpm(psd, frequency, buffer_size, sampling_rate, tr=2):
    #frequency = scipy.fft.rfftfreq(n=5 * buffer_size, d=(1 / sampling_rate))
    frequency *= 60
    #print('frequency : ', frequency)
    fst_max_idx = np.argmax(psd)
    wo_max_PSD_pair_PG = psd.copy()
    wo_max_PSD_pair_PG[fst_max_idx] = np.min(psd)
    fst_max = np.max(psd)
    sec_max = np.max(wo_max_PSD_pair_PG)

    #v1 = frequency[fst_max_idx]
    #print('v1 : ', v1)

    if fst_max/sec_max > tr:
        return round(limit_bpm(psd, frequency, 144))
        #return round(frequency[fst_max_idx])
    else:
        return None


def limit_bpm(PSD, Frequency, limit):
    idx = np.argmax(PSD)
    bpm = Frequency[idx]
    #print("select bpm : ", bpm)
    if bpm > limit:
        wo_max_PSD = PSD.copy()
        wo_max_PSD[idx] = np.min(PSD)
        bpm = limit_bpm(wo_max_PSD, Frequency, limit)
    return bpm

def select_pg_signal(raw_signal, ica_signal):
    max_dis = 0
    max_dis_index = 0
    mean = [None]*raw_signal.shape[0]
    for i in np.arange(raw_signal.shape[0]):
        mean[i] = np.mean(raw_signal[i])
    avg_mean = np.mean(mean)

    for i in np.arange(ica_signal.shape[0]):
        temp = math.sqrt(pow((np.mean(ica_signal[i]) - avg_mean), 2))
        if temp > max_dis:
            max_dis = temp
            max_dis_index = i

    return ica_signal[max_dis_index]



