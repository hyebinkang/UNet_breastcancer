import tensorflow as tf
import os
import numpy as np
import random

from tqdm import tqdm                           #프로그램이 의도한 것 처럼 돌아가고 있는지 작업 확인 모듈(progress bar)

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = 'stage1_train/'            #folder-첫번째 객체[0]
TEST_PATH = 'stage1_test/'

seed = 42                                       #특정 시작 숫자값(seed)을 정해주면 난수처럼 보이는 수열 생성
np.random.seed = seed

train_ids = next(os.walk(TRAIN_PATH))[1]                 #하위 directory folder에 차례대로 접근, next: 반복 가능한 타입의 객체, sub folder[1]-> 다음 객체를 적어준다(두번째 객체)
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)               #0벡터로 공간 만들어 주기(train 개수만큼(670), 128*128*3)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)                           #test, mask

print("resizing training images and masks")
#train images
print(len(TRAIN_PATH), len(TEST_PATH))
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):                                     #enumerate
    path = TRAIN_PATH + id_                                                                         #이미지에 접근
    img = imread(path+'/images/'+ id_+ '.png')[:,:,:IMG_CHANNELS]

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img                                                                                #empty array(X_train) 채우기

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)                                      #0벡터로 마스크 파일 공간 만들어주기, channel = 1

    for mask_file in next(os.walk(path+'/masks/'))[2]:                                              #sub folder[2]
        mask_ = imread(path+'/masks/'+mask_file)                                                    #mask폴더에 접근
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)

    Y_train[n] = mask                                                                               #empty array(Y_train)채우기

#test images, mask 사진 없음
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode= 'constant', preserve_range=True)
    X_test[n] = img

print('Done')
# image_x = random.randint(0, len(TRAIN_PATH))              #이미지 출력
# imshow(X_train[image_x])
# plt.show()
# imshow(np.squeeze(Y_train[image_x]))
# plt.show()

# build the model
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)        # int->float로 바꾸어줘야 함, lambda를 사용해서 x: x/255 넣기

# Contraction Path
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)        # 128*128*16
print(c1.shape)
c1 = tf.keras.layers.Dropout(0.1)(c1)
print(c1.shape)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
print(c1.shape)
p1 = tf.keras.layers.MaxPooling2D((2,2), strides=2)(c1)                                                             # 64*64*16
print(p1.shape)

c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)       # 64*64*32
print(c2.shape)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2), strides= 2)(p1)                                                            # 32*32*32

c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)       # 32*32*64
c3 = tf.keras.layers.Dropout(0.1)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2), strides= 2)(p2)                                                            #16*16*64

c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)      #16*16*128
c4 = tf.keras.layers.Dropout(0.1)(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2), strides= 2)(p3)                                                            #8*8*128

c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)      #8*8*256
c5 = tf.keras.layers.Dropout(0.1)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)      #8*8*256

#expansive path

u6 = tf.keras.layers.Conv2DTranspose(128,(2,2), strides=(2,2), padding='same')(c5)                                  #16*16*256 -> 16*16*128
u6 = tf.keras.layers.concatenate([u6, c4])                                                                          #16*16*128 concat 16*16*128 = 16*16*256
c6 = tf.keras.layers.Conv2D(128,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)       #16*16*128

u7 = tf.keras.layers.Conv2DTranspose(64,(2,2), strides=(2,2), padding='same')(c6)                                    #32*32*128->32*32*64
u7 = tf.keras.layers.concatenate([u7, c3])                                                                          #32*32*64 concat 32*32*64 = 32*32*128
c7 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)        #32*32*64

u8 = tf.keras.layers.Conv2DTranspose(32,(2,2), strides=(2,2), padding='same')(c7)                                    #64*64*64->64*64*32
u8 = tf.keras.layers.concatenate([u8, c2])                                                                          #64*64*32 concat 64*64*32
c8 = tf.keras.layers.Conv2D(32,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.2)(c8)
c8 = tf.keras.layers.Conv2D(32,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)        #64*64*32

u9 = tf.keras.layers.Conv2DTranspose(16,(2,2), strides=(2,2), padding='same')(c8)                                    #128*128*32 -> 128*128*16
u9 = tf.keras.layers.concatenate([u9, c1])                                                                   #128*128*16 concat 128*128*16 = 128*128*32
c9 = tf.keras.layers.Conv2D(16,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.2)(c9)
c9 = tf.keras.layers.Conv2D(16,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)        #128*128*16

outputs = tf.keras.layers.Conv2D(1,(1,1), activation='sigmoid')(c9)                                                  #128*128*1

model = tf.keras.Model(inputs = [inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

######################################
#Model Check point

checkpointer = tf.keras.callbacks.ModelCheckpoint('unet_checkpoint.h5', verbose=1, save_best_only=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

results = model.fit(X_train,Y_train, validation_split=0.1, batch_size=16, epochs=100, callbacks=callbacks)

######################################
#perform a sanity check on some random training samples

idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)                     #670*0.9
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)                       #670*0.9
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)                                            #threshold
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test = (preds_test > 0.5).astype(np.uint8)

#perform a sanity check on some random training samples
ix = np.random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))                                                                 #squeeze 길이가 1인 축을 제거
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

#perform a sanity check on some random validation samples
ix = np.random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()