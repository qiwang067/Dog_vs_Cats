
# coding: utf-8

#数据集整理
import os,shutil
#原始数据集目录
original_dataset_dir = 'data/kaggle_original_data'

#保存较小数据集的目录
base_dir = 'data/cats_and_dogs_small'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

#分别对应划分后的训练、验证和测试的目录
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
    
# 猫的训练图像目录
train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)

# 狗的训练图像目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)

validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
    
#猫的验证图像目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_cats_dir):
    os.mkdir(validation_cats_dir)

# 狗的验证图像目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)
    
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
    
# 猫的测试图像目录
test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)

# 狗的测试图像目录
test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)
    
# 将前1000张猫的图像复制到 train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# 将接下来 500 张猫的图像复制到 validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# 将接下来的 500 张猫的图像复制到 test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# 将前 1000 张狗的图像复制到 train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# 将接下来 500 张狗的图像复制到 validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# 将接下来 500 张狗的图像复制到 test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

#查看每个分组中（训练/验证/测试）中分别包含多少张图像
print('total training cat images:',len(os.listdir(train_cats_dir)))
print('total training dog images:',len(os.listdir(train_dogs_dir)))
print('total validation cat images:',len(os.listdir(validation_cats_dir)))
print('total validation dog images:',len(os.listdir(validation_dogs_dir)))
print('total test cat images:',len(os.listdir(test_cats_dir)))
print('total test dog images:',len(os.listdir(test_dogs_dir)))




#建立模型
from keras.applications import VGG16
#weghts指定模型初始化的权重检查点，include_top指定模型的最后是否包含密集连接分类器
# input_shape是输入到网络中图像张量的形状,这个参数是可选的，如果不传
#入这个参数，那么网络能够处理任意形状的输入。
conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
from keras import layers
from keras import models
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
#查看特征图的维度随每层的变化
model.summary()
#冻结直到某一层的所有层
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1'
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = Flase




#训练模型
from keras.preprocessing.image import ImageDataGenerator
#数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,    #将图像乘以1/255缩放
    rotation_range=40, #图片随机旋转40度
    width_shift_range=0.2, #图片在水平方向上平移的距离为图片的总宽度的0.2倍
    height_shift_range=0.2, #图片在垂直方向上平移的距离为图片的总高度的0.2倍
    shear_range=0.2,  #随机错切变换的角度
    zoom_range=0.2,  #图像随机缩放的范围
    horizontal_flip=True,
    fill_model='nearest') #随机将一半图像翻转

test_datagen = ImageDataGenerator(rescale=1./255)  #不能增强验证数据

#数据预处理
train_generator = train_datagen.flow_from_directory(
    train_dir,  #目标目录
    target_size = (150,150), #将所有图像的大小调整为150×150
    batch_size = 20,
    class_mode = 'binary')  #因为使用binary_crossentropy损失，所以用二进制标签

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary')

#编译模型
from keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=e-5),
              metrics=['acc'])

#利用批量生成器拟合模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps = 50)

#保存模型
model.save('cats_and_dogs.h5')




#绘制模型在训练集和验证集上的表现
import matplotlib.pyplot as plt
def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()




#在测试集上评估模型
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)

