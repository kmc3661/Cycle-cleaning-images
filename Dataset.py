import os
import matplotlib.pyplot as plt
from models.cycleGAN import CycleGAN
from utils.loaders import DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# run params

DATA_NAME = 'raindrop2clean'
RUN_FOLDER = 'result/'
RUN_FOLDER += DATA_NAME


if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER,'viz'))
    os.mkdir(os.path.join(RUN_FOLDER,'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode='build'

IMAGE_SIZE=256
data_loader=DataLoader(dataset_name=DATA_NAME,img_res=(IMAGE_SIZE,IMAGE_SIZE))

gan=CycleGAN(
    input_dim=(IMAGE_SIZE,IMAGE_SIZE,3)
    ,learning_rate=0.0002
    ,buffer_max_length=50
    ,lambda_validation=1 #gan_loss의 가중치
    ,lambda_reconstr=10#cycle_loss에 대한 가중치
    ,lambda_id=2 #identity_loss에 대한 가중치
    ,generator_type='resnet'
    ,gen_n_filters=32
    ,disc_n_filters=32)

if mode=='build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(os.path.join(RUN_FOLDER,'weights/weights.h5')) #build mode가 아닐때 weight파일 불러오기

#plot
gan.g_BA.summary()# B->A domain generator
gan.g_AB.summary()# A->B domain generator
gan.d_A.summary()# A가 real인지 아닌지 판별하는 discriminator
gan.d_B.summary()#B ''

#train
BATCH_SIZE=1
EPOCHS=120
PRINT_EVERY_N_BTACHES=100 #10마다 모델을 저장

TEST_A_FILE='n07740461_14740.jpg'
TEST_B_FILE = 'n07749192_4241.jpg'

gan.train(data_loader
          ,run_folder=RUN_FOLDER
          ,epochs=EPOCHS
          ,test_A_file=TEST_A_FILE
          ,test_B_file=TEST_B_FILE
          ,batch_size=BATCH_SIZE
          ,sample_interval=PRINT_EVERY_N_BTACHES) #train함수 호출하여 학습

#loss visualization

fig=plt.figure(figsize=(20,10))

plt.plot([x[1] for x in gan.g_losses], color='green', linewidth=0.1) #DISCRIM LOSS
# plt.plot([x[2] for x in gan.g_losses], color='orange', linewidth=0.1)
plt.plot([x[3] for x in gan.g_losses], color='blue', linewidth=0.1) #CYCLE LOSS
# plt.plot([x[4] for x in gan.g_losses], color='orange', linewidth=0.25)
plt.plot([x[5] for x in gan.g_losses], color='red', linewidth=0.25) #ID LOSS
# plt.plot([x[6] for x in gan.g_losses], color='orange', linewidth=0.25)

plt.plot([x[0] for x in gan.g_losses], color='black', linewidth=0.25)

# plt.plot([x[0] for x in gan.d_losses], color='black', linewidth=0.25)

plt.xlabel('batch',fontsize=18)
plt.ylabel('loss',fontsize=16)

plt.ylim(0,5)

plt.show()

