import sys
import os
LOCAL_PACKAGE_DIR = os.path.abspath("./keras-yolo3")
sys.path.append(LOCAL_PACKAGE_DIR)

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data 
# get_random_data - Image Augmentation 하기 위한 모듈 - Augmentation을 할 때 Bounding Box 처리도 해준다.

import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from train import get_classes, get_anchors
from train import create_model, data_generator, data_generator_wrapper

#gpu oom 문제
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 0.6 sometimes works better for folks
K.tensorflow_backend.set_session(tf.Session(config=config))

BASE_DIR = 'C:/Users/kate1/capstone/keras-yolo3/'
ANNO_DIR = 'C:/Users/kate1/capstone/train_image/annotation/'

## 학습을 위한 기반 환경 설정. annotation 파일 위치, epochs시 저장된 모델 파일, Object클래스 파일, anchor 파일.
annotation_path = os.path.join(ANNO_DIR, 'helmet_anno.csv')
log_dir = os.path.join(BASE_DIR, 'log_dir/')  # tensorboard, epoch 당 weight 저장 장소로 사용할 될 예정이다.
classes_path = os.path.join(BASE_DIR, 'model_data/helmet_classes.txt')    # 내가 만듬 한줄 - raccoon
# tiny yolo로 모델을 학습 원할 시 아래를 yolo_anchors.txt' -> tiny_yolo_anchors.txt'로 수정. 
anchors_path = os.path.join(BASE_DIR,'model_data/yolo_anchors.txt')      # 그대로 사용

class_names = get_classes(classes_path)
num_classes = len(class_names)   # 1개
print(num_classes)
anchors = get_anchors(anchors_path)

# 아래는 원본 train.py에서 weights_path 변경을 위해 임의 수정. 최초 weight 모델 로딩은 coco로 pretrained된 모델 로딩. 
# tiny yolo로 모델을 학습 원할 시 아래를 model_data/yolo.h5' -> model_data/tiny-yolo.h5'로 수정. 
model_weights_path = os.path.join(BASE_DIR, 'model_data/yolo.h5' )

input_shape = (416,416) # yolo-3 416 을 사용하므로 이렇게 정의. 라쿤 이미지들 wh는 모두 다르다.

is_tiny_version = len(anchors)==6 # default setting
# create_tiny_model(), create_model() 함수의 인자 설정을 원본 train.py에서 수정. 
if is_tiny_version:
    model = create_tiny_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path=model_weights_path)
else:
    # create_model 은 해당 패키지의 tarin.py 내부에 있는 클래스를 사용했다. 이 함수는 keras 모듈이 많이 사용한다. 우선 모르는 건 pass하고 넘어가자.
    model = create_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path=model_weights_path) # make sure you know what you freeze

# epoch 마다 call back 하여 모델 파일 저장.
# 이것 또한 Keras에서 많이 사용하는 checkpoint 저장 방식인듯 하다. 우선 이것도 모르지만 넘어가자.
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

val_split = 0.1  # train data : val_data = 9 : 1

with open(annotation_path) as f:
    # 이러니 annotation 파일이 txt이든 csv이든 상관없었다.
    lines = f.readlines()

# 랜덤 시드 생성 및 lines 셔플하기
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)

# 데이터셋 나누기
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val

# 여기서 부터 진짜 학습 시작! 
# create_model() 로 반환된 yolo모델에서 trainable=False로 되어 있는 layer들 제외하고 학습
# if True:
#     # optimizer와 loss 함수 정의
#     # 위에서 사용한 create_model 클래스의 맴버함수를 사용한다. 
#     model.compile(optimizer=Adam(lr=1e-3), loss={
#         # use custom yolo_loss Lambda layer.
#         'yolo_loss': lambda y_true, y_pred: y_pred})

#     batch_size = 4
#     print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
#     # foward -> backpropagation -> weight 갱신 -> weight 저장
#     # checkpoint 만드는 것은 뭔지 모르겠으니 pass...
#     model.fit_generator(
#             data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
#             steps_per_epoch=max(1, num_train//batch_size),
#             validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
#             validation_steps=max(1, num_val//batch_size),
#             epochs=50,
#             initial_epoch=0,
#             callbacks=[logging, checkpoint])
#     model.save_weights(log_dir + 'trained_weights_stage_1.h5')

# create_model() 로 반환된 yolo모델에서 trainable=False로 되어 있는 layer들 없이, 모두 True로 만들고 다시 학습
if True:
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    print('Unfreeze all of the layers.')

    batch_size = 4 # note that more GPU memory is required after unfreezing the body
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(
        data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=100,
        initial_epoch=50,
        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'helmet_weights_final.h5')