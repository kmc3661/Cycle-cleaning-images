from __future__ import print_function, division
import scipy

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.merge import add
from models.layers.layers import ReflectionPadding2D
from keras.models import Sequential, Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K
from PIL import Image
import cv2



from keras.utils import plot_model

import datetime
import matplotlib.pyplot as plt
import sys

import numpy as np
import os
import pickle as pkl
import random

from collections import deque


class CycleGAN():
    def __init__(self
        , input_dim
        , learning_rate
        , lambda_validation
        , lambda_reconstr
        , lambda_id
        , generator_type
        , gen_n_filters
        , disc_n_filters
        , buffer_max_length = 50
        ):

        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.buffer_max_length = buffer_max_length
        self.lambda_validation = lambda_validation
        self.lambda_reconstr = lambda_reconstr
        self.lambda_id = lambda_id
        self.generator_type = generator_type
        self.gen_n_filters = gen_n_filters
        self.disc_n_filters = disc_n_filters

        # Input shape
        self.img_rows = input_dim[0]
        self.img_cols = input_dim[1]
        self.channels = input_dim[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0

        self.buffer_A = deque(maxlen = self.buffer_max_length)#deque: 양방향 큐로 append, pop이 매우 빠름
        self.buffer_B = deque(maxlen = self.buffer_max_length)
        
        # Calculate output shape of D (PatchGAN)-> image별이 아닌 patch별로 진위여부 구분
        patch = int(self.img_rows / 2**3)
        self.disc_patch = (patch, patch, 1)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)#평균은 0 표준편차 0.02로 초기화

        self.compile_models()

        
    def compile_models(self):

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()#A domain 진위 판별
        self.d_B = self.build_discriminator()#B domain 진위 판별
        
        self.d_A.compile(loss='mse',
            optimizer=Adam(self.learning_rate, 0.5),
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=Adam(self.learning_rate, 0.5),
            metrics=['accuracy'])


        # Build the generators
        if self.generator_type == 'unet':
            self.g_AB = self.build_generator_unet()
            self.g_BA = self.build_generator_unet()
        else:
            self.g_AB = self.build_generator_resnet()
            self.g_BA = self.build_generator_resnet()

        # For the combined model we will only train the generators
        self.d_A.trainable = False# -> discriminaotor와 generator는 따로 학습시켜줘야 하기때문
        self.d_B.trainable = False

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain->  for checking Cycle loss
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'], #mae: Mean Absolute Error(오차의 절대값에 대한 평균, 즉 L1 Loss), mse: Mean squared Error
                            loss_weights=[  self.lambda_validation,                       self.lambda_validation,
                                            self.lambda_reconstr, self.lambda_reconstr,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=Adam(0.0002, 0.5))

        self.d_A.trainable = True#generator의 학습이 끝났으므로 discriminator를 다시 학습
        self.d_B.trainable = True
    

    def build_generator_unet(self):

        def downsample(layer_input, filters, f_size=4):#input_size(128,128,3)
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = InstanceNormalization(axis = -1, center = False, scale = False)(d)
            d = Activation('relu')(d)
            
            return d

        def upsample(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
            u = InstanceNormalization(axis = -1, center = False, scale = False)(u)
            u = Activation('relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)

            u = Concatenate()([u, skip_input])
            return u

        # Image input
        img = Input(shape=self.img_shape)

        # Downsampling
        d1 = downsample(img, self.gen_n_filters) 
        d2 = downsample(d1, self.gen_n_filters*2)
        d3 = downsample(d2, self.gen_n_filters*4)
        d4 = downsample(d3, self.gen_n_filters*8)

        # Upsampling
        u1 = upsample(d4, d3, self.gen_n_filters*4)
        u2 = upsample(u1, d2, self.gen_n_filters*2)
        u3 = upsample(u2, d1, self.gen_n_filters)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(img, output_img)


    def build_generator_resnet(self):

        def conv7s1(layer_input, filters, final):
            y = ReflectionPadding2D(padding =(3,3))(layer_input)
            y = Conv2D(filters, kernel_size=(7,7), strides=1, padding='valid', kernel_initializer = self.weight_init)(y)
            # kernel의 초기값은 정규분포(평균:0, 표준편차:0.2)
            if final:
                y = Activation('tanh')(y)#마지막 Conv block일때는 출력을 -1~1 사이의 값을 위해 tanh 사용
            else:
                y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
                # axis=-1 :  평균과 표준편차를 구해서 sample별로 각 채널의 정규화를 진행하여 원본 이미지와 tranfer된 이미지를 구분될수 없도록 만듬
                # BatchNormalization:  전체 데이터를 같은 평균과 분산으로 정리,
                # instanceNormalization: 하나 하나의 이미지를 자신들의 평균과 분산으로 정리
                y = Activation('relu')(y)
            return y

        def downsample(layer_input,filters):
            y = Conv2D(filters, kernel_size=(3,3), strides=2, padding='same', kernel_initializer = self.weight_init)(layer_input)
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
            y = Activation('relu')(y)
            return y

        def residual(layer_input, filters):
            shortcut = layer_input
            y = ReflectionPadding2D(padding =(1,1))(layer_input) # (34,34,128)
            y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='valid', kernel_initializer = self.weight_init)(y)#(32,32,128)
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
            y = Activation('relu')(y)
            
            y = ReflectionPadding2D(padding =(1,1))(y)
            y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='valid', kernel_initializer = self.weight_init)(y)
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)#(32,32,128)

            return add([shortcut, y])

        def upsample(layer_input,filters):
            y = Conv2DTranspose(filters, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer = self.weight_init)(layer_input)
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)  #Transposed Convolution은 학습할수있는 upsampling, input에 변형이 가해짐
            y = Activation('relu')(y)
    
            return y


        # Image input
        img = Input(shape=self.img_shape)#(256,256,3)

        y = img

        y = conv7s1(y, self.gen_n_filters, False)#(256,256,32)
        y = downsample(y, self.gen_n_filters * 2)  # (128,128,64)
        y = downsample(y, self.gen_n_filters * 4)#(64,64,128)
        y = downsample(y, self.gen_n_filters * 8)#(32,32,256)
        y = residual(y, self.gen_n_filters * 8)
        y = residual(y, self.gen_n_filters * 8)
        y = residual(y, self.gen_n_filters * 8)
        y = residual(y, self.gen_n_filters * 8)
        y = residual(y, self.gen_n_filters * 8)
        y = residual(y, self.gen_n_filters * 8)
        y = residual(y, self.gen_n_filters * 8)
        y = residual(y, self.gen_n_filters * 8)
        y = residual(y, self.gen_n_filters * 8)#(32,32,256)
        y = upsample(y, self.gen_n_filters * 4)  # (64,64,128)
        y = upsample(y, self.gen_n_filters * 2)#(128,128,64)
        y = upsample(y, self.gen_n_filters)#(256,256,32)
        y = conv7s1(y, 3, True)#(Tanh function을 사용), (256,256,3)
        output = y#(256,256,3)

   
        return Model(img, output)


    def build_discriminator(self):

        def conv4(layer_input,filters, stride = 2, norm=True):
            y = Conv2D(filters, kernel_size=(4,4), strides=stride, padding='same', kernel_initializer = self.weight_init)(layer_input)
            
            if norm:
                y = InstanceNormalization(axis = -1, center = False, scale = False)(y)

            y = LeakyReLU(0.2)(y)
           
            return y

        img = Input(shape=self.img_shape)#(256,256,3)

        y = conv4(img, self.disc_n_filters, stride = 2, norm = False)#(128,128,32)
        y = conv4(y, self.disc_n_filters*2, stride = 2)#(64,64,64)
        y = conv4(y, self.disc_n_filters*4, stride = 2)#(32,32,128)
        y = conv4(y, self.disc_n_filters*8, stride = 1)#(32,32,256)

        output = Conv2D(1, kernel_size=4, strides=1, padding='same',kernel_initializer = self.weight_init)(y)#(32,32,1)

        return Model(img, output)

    def train_discriminators(self, imgs_A, imgs_B, valid, fake):

        # Translate images to opposite domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)

        self.buffer_B.append(fake_B)#학습의 안정성을 위해 다시 dicriminator에 보여주기위해 버퍼에 추가
        self.buffer_A.append(fake_A)

        fake_A_rnd = random.sample(self.buffer_A, min(len(self.buffer_A), len(imgs_A)))#버퍼에 있는것을 샘플링
        fake_B_rnd = random.sample(self.buffer_B, min(len(self.buffer_B), len(imgs_B)))

        # Train the discriminators (original images = real / translated = Fake)
        dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)#valid: 진짜 같아 보이는지
        dA_loss_fake = self.d_A.train_on_batch(fake_A_rnd, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
        # fit: 전체 이미지 데이터로부터 배치단위로 학습
        # train_on_batch: 배치크기의 데이터만 입력받고 1회만 학습, 0번째 요소엔 loss 1번째 요소엔 Accuraccy를 return
        # -> 한 이미지에 대해서 각각 학습이 일어나게되므로 Gan에서 많이 사용됨

        dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
        dB_loss_fake = self.d_B.train_on_batch(fake_B_rnd, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss_total = 0.5 * np.add(dA_loss, dB_loss)

        return (
            d_loss_total[0]
            , dA_loss[0], dA_loss_real[0], dA_loss_fake[0]
            , dB_loss[0], dB_loss_real[0], dB_loss_fake[0]
            , d_loss_total[1]
            , dA_loss[1], dA_loss_real[1], dA_loss_fake[1]
            , dB_loss[1], dB_loss_real[1], dB_loss_fake[1]
        )

    def train_generators(self, imgs_A, imgs_B, valid):

        # Train the generators
        # self.combined=Model(inputs=[img_A, img_B],
        # output=[ valid_A, valid_B, -> generated image가 얼마나 유사한지
        # reconstr_A, reconstr_B,-> 원래의 도메인으로  복원한 이미지가 원래 이미지와 얼마나 유사한지(Cycled Loss)
        # img_A_id,img_B_id])-> identity loss
        return self.combined.train_on_batch([imgs_A, imgs_B],
                                                [valid, valid,
                                                imgs_A, imgs_B,
                                                imgs_A, imgs_B])


    def train(self, data_loader, run_folder, epochs, test_A_file, test_B_file, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(self.epoch, epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch()):

                d_loss = self.train_discriminators(imgs_A, imgs_B, valid, fake)
                g_loss = self.train_generators(imgs_A, imgs_B, valid)

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                    % ( self.epoch, epochs,
                        batch_i, data_loader.n_batches,
                        d_loss[0], 100*d_loss[7],
                        g_loss[0],
                        np.sum(g_loss[1:3]),
                        np.sum(g_loss[3:5]),
                        np.sum(g_loss[5:7]),
                        elapsed_time))

                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(data_loader, batch_i, run_folder)
                    self.combined.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (self.epoch)))
                    self.combined.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                    self.save_model(run_folder)

                
            self.epoch += 1

    # def sample_images(self, data_loader, batch_i, run_folder):
    #
    #     r, c = 2, 4
    #
    #     imgs_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
    #     imgs_B = data_loader.load_data(domain="B", batch_size=1, is_testing=True)
    #     # image = cv2.imread('D:/Temp/GDL_code-master/data/raindrop2clean/testA/rain13.jpg')
    #     # fake_B=self.g_AB.predict(image)
    #     # gen_imgs=image
    #
    #
    #    # Translate images to the other domain
    #     fake_B = self.g_AB.predict(imgs_A)
    #     fake_A = self.g_BA.predict(imgs_B)
    #
    #     # Translate back to original domain
    #     reconstr_A = self.g_BA.predict(fake_B)
    #     reconstr_B = self.g_AB.predict(fake_A)
    #
    #     # ID the images
    #     id_A = self.g_BA.predict(imgs_A)
    #     id_B = self.g_AB.predict(imgs_B)
    #
    #     gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, id_A, imgs_B, fake_A, reconstr_B, id_B])
    #
    #     # Rescale images 0 - 1
    #     # gen_imgs=127.5*np.array(gen_imgs)+127.5
    #     # gen_imgs=np.clip(gen_imgs,0,255)
    #     # img = Image.fromarray(gen_imgs)
    #     # img.save(os.path.join(run_folder, "images/%d_%d.jpg" % ( self.epoch, batch_i)))
    #
    #     gen_imgs = 0.5 * gen_imgs + 0.5
    #     gen_imgs = np.clip(gen_imgs, 0, 1)
    #
    #     titles = ['Original', 'Translated', 'Reconstructed', 'ID']
    #     fig, axs = plt.subplots(r, c, figsize=(25, 12.5))
    #     cnt = 0
    #     for i in range(r):
    #         for j in range(c):
    #             axs[i, j].imshow(gen_imgs[cnt])
    #             axs[i, j].set_title(titles[j])
    #             axs[i, j].axis('off')
    #             cnt += 1
    #     fig.savefig(os.path.join(run_folder, "images/%d_%d.png" % ( self.epoch, batch_i)))
    #     plt.close()

    def sample_images(self, data_loader, batch_i, run_folder):

        imgs_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)

       # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)

        # Rescale images 0 - 1
        gen_imgs = np.array(127.5 * fake_B + 127.5)
        gen_imgs = gen_imgs.astype(int)
        gen_imgs = np.clip(gen_imgs, 0, 255)
        gen_imgs = gen_imgs.reshape(256, 256, 3)
        plt.imsave(os.path.join(run_folder, "images/%d_%d.png" % (self.epoch, batch_i)), gen_imgs)
        # img = Image.fromarray(gen_imgs, 'RGB')
        # img.save(os.path.join(run_folder, "images/%d_%d.png" % (self.epoch, batch_i)))


    def plot_model(self, run_folder):
        plot_model(self.combined, to_file=os.path.join(run_folder ,'viz/combined.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.d_A, to_file=os.path.join(run_folder ,'viz/d_A.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.d_B, to_file=os.path.join(run_folder ,'viz/d_B.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.g_BA, to_file=os.path.join(run_folder ,'viz/g_BA.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.g_AB, to_file=os.path.join(run_folder ,'viz/g_AB.png'), show_shapes = True, show_layer_names = True)


    def save(self, folder):

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim
                ,  self.learning_rate
                ,  self.buffer_max_length
                ,  self.lambda_validation
                ,  self.lambda_reconstr
                ,  self.lambda_id
                ,  self.generator_type
                ,  self.gen_n_filters
                ,  self.disc_n_filters
                ], f)# pickle.dump(list,file): list 내용을 file에 입력

        self.plot_model(folder)


    def save_model(self, run_folder):


        self.combined.save(os.path.join(run_folder, 'model.h5')  )
        self.d_A.save(os.path.join(run_folder, 'd_A.h5') )
        self.d_B.save(os.path.join(run_folder, 'd_B.h5') )
        self.g_BA.save(os.path.join(run_folder, 'g_BA.h5')  )
        self.g_AB.save(os.path.join(run_folder, 'g_AB.h5') )

        pkl.dump(self, open( os.path.join(run_folder, "obj.pkl"), "wb" ))

    def load_weights(self, filepath):
        self.combined.load_weights(filepath)
