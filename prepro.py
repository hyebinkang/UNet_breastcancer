import numpy as np
import cv2
import os

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

TRAIN_DIR = os.getcwd() + '/benign/'
TRAIN_PATH = os.listdir('benign/')
TEST_DIR = os.getcwd() + '/malignant/'
TEST_PATH = os.listdir('malignant/')

# def load_data(mode=False):
#     if mode == 'train':
#         path_away = TRAIN_PATH
#         directory = TRAIN_DIR
#     elif mode == 'test':
#         path_away = TEST_PATH
#         directory = TEST_DIR
#
#     mask_list = []
#     ori_list = []
#
#     for i in path_away:
#         if 'mask' in i:
#             mask_list.append(i)
#         else:
#             ori_list.append(i)
#
#     mask = np.zeros((len(ori_list), IMG_WIDTH, IMG_HEIGHT))
#     origin = np.zeros((len(ori_list), IMG_WIDTH, IMG_HEIGHT))
#
#     for n, i in enumerate(ori_list):
#         img = cv2.imread(directory+i, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
#         origin[n] = img/255.0
#
#     for n, i in enumerate(ori_list):
#         lists = list(filter(lambda x: i.split('.')[0] in x, mask_list))
#         new_img = np.zeros((IMG_WIDTH, IMG_HEIGHT))
#         for j in lists:
#             img = cv2.imread(directory + j, cv2.IMREAD_GRAYSCALE)
#             img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
#             new_img += img
#         new_img = new_img / 255.0
#         new_img[new_img >= 1] = 1
#         mask[n] = new_img
#
#     origin = origin[:, :, :, np.newaxis]
#     mask = mask[:, :, :, np.newaxis]
#
#     return origin, mask

def load_data():

    mask_list = []                  #마스크 이미지 목록 리스트
    ori_list = []                   #원본 이미지 목록 리스트

    for i in TRAIN_PATH:
        if 'mask' in i:
            mask_list.append(i)     #마스크 이미지
        else:
            ori_list.append(i)      #원본 이미지

    mask = np.zeros((len(ori_list), IMG_HEIGHT, IMG_WIDTH))                 #제로벡터 사용하여 원본 이미지 길이 만큼 공간 만들기
    origin = np.zeros((len(ori_list), IMG_HEIGHT, IMG_WIDTH))

    for n, i in enumerate(ori_list):                                        #enumerate 사용(n: 인덱스 번호, i: 값)
        img = cv2.imread(TRAIN_DIR + i, cv2.IMREAD_GRAYSCALE)               #grayscale로 가져와서 읽기(원본 width, 원본 heigth)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))                      #resize로 길이 조절
        origin[n] = img / 255.0                                             #정규화

    test_ori = []
    for i in TEST_PATH:
        if 'mask' not in i:
            test_ori.append(i)                                              # test 이미지 내의 원본 이미지만을 사용

    test_ori_list = np.zeros((len(test_ori), IMG_HEIGHT, IMG_WIDTH))        # 제로벡터로 이미지가 들어갈 공간 생성

    for n,i in enumerate(test_ori):
        img = cv2.imread(TEST_DIR+i, cv2.IMREAD_GRAYSCALE)                  # test이미지 가져와서 resize 및 정규화
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        test_ori_list[n] = img/255.0


    for n, i in enumerate(ori_list):
        lists = list(filter(lambda x: i.split('.')[0] in x, mask_list))     #.을 기준으로 x와 공통된 부분이 i 사진 이름에 들어가면 같은 리스트로 만들기

        new_img = np.zeros((IMG_HEIGHT, IMG_HEIGHT))                        #한 행에 들어갈 128*128만 먼저 정해주기
        for j in lists:
            img = cv2.imread(TRAIN_DIR + j, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            new_img += img                                                  #128*128에 맞춰서 이미지 더해주기
        new_img = new_img / 255.0
        new_img[new_img >= 1] = 1                                           #이미지에 겹치는 부분이 있다면 1로 맞추기
        mask[n] = new_img                                                   #mask_list에 합친 이미지 넣기

    origin = origin[:, :, :, np.newaxis]                                #B,H,W,C로 축 증가
    mask = mask[:, :, :, np.newaxis]
    test_ori_list = test_ori_list[:,:,:, np.newaxis]
    print(origin.shape)

    return origin, mask, test_ori_list