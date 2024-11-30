from tensorflow import keras
from keras import losses
from keras import layers, Input, Model

# class Inception(keras.Model):
#     def __init__(self, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool):
#         super(Inception, self).__init__()
# 
#         self.path_1 = layers.Conv2D(filters_1x1, kernel_size=(1, 1), padding='same', activation='relu')
#         self.path_2 = Sequential([
#             layers.Conv2D(filters_3x3_reduce, kernel_size=(1, 1), padding='same', activation='relu'),
#             layers.Conv2D(filters_3x3, kernel_size=(3, 3), padding='same', activation='relu')
#         ])
#         self.path_3 = Sequential([
#             layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu'),
#             layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')
#         ])
#         self.path_4 = Sequential([
#             layers.MaxPool2D((3, 3), strides=(1, 1), padding='same'),
#             layers.Conv2D(filters_pool, (1, 1), padding='same', activation='relu')
#         ])
# 
#     def call(self, x):
#         path1 = self.path_1(x)
#         path2 = self.path_2(x)
#         path3 = self.path_3(x)
#         path4 = self.path_4(x)
#         return layers.concatenate(inputs=[path1, path2, path3, path4], axis=3)
# 
# class Googlenet(keras.Model):
#     def __init__(self):
#         super(Googlenet, self).__init__()
# 
#         self.sequential_1 = Sequential([
#             layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation='relu'),
#             layers.MaxPooling2D(3, strides=2),
#             layers.Conv2D(64, 1, strides=1, padding='same', activation='relu'),
#             layers.Conv2D(192, 3, strides=1, padding='same', activation='relu'),
#             layers.MaxPooling2D(3, strides=2),
#             Inception(filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,
#                       filters_5x5_reduce=16, filters_5x5=32, filters_pool=32),
#             Inception(filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192,
#                       filters_5x5_reduce=32, filters_5x5=96, filters_pool=64),
#             layers.MaxPooling2D(3, strides=2),
#             Inception(filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208,
#                       filters_5x5_reduce=16, filters_5x5=48, filters_pool=64),
#         ])
#         self.aux_1 = Sequential([
#             layers.AveragePooling2D((5, 5), strides=3),
#             layers.Conv2D(128, 1, padding='same', activation='relu'),
#             layers.Flatten(),
#             layers.Dense(1024, activation='relu'),
#             layers.Dropout(0.7),
#             layers.Dense(4, activation='softmax'),
#         ])
#         self.sequential_2 = Sequential([
#             Inception(filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24, filters_5x5=64, filters_pool=64),
#             Inception(filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24, filters_5x5=64, filters_pool=64),
#             Inception(filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32, filters_5x5=64, filters_pool=64)
#         ])
#         self.aux_2 = Sequential([
#             layers.AveragePooling2D((5, 5), strides=3),
#             layers.Conv2D(128, 1, padding='same', activation='relu'),
#             layers.Flatten(),
#             layers.Dense(1024, activation='relu'),
#             layers.Dropout(0.7),
#             layers.Dense(4, activation='softmax')
#         ])
#         self.aux_3 = Sequential([
#             Inception(filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool=128),
#             layers.MaxPooling2D(3, strides=2),
#             Inception(filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool=128),
#             Inception(filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5_reduce=48, filters_5x5=128, filters_pool=128),
#             layers.GlobalAveragePooling2D(),
#             layers.Dropout(0.4),
#             layers.Dense(4, activation='softmax')
#         ])
#     def call(self, x):
#         x       = self.sequential_1(x)
#         aux_1   = self.aux_1(x)
#         x       = self.sequential_2(x)
#         aux_2   = self.aux_2(x)
#         x       = self.aux_3(x)
#         return {'result':x, 'aux_1':aux_1, 'aux_2':aux_2}
def inception(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool):

    path1 = layers.Conv2D(filters_1x1, kernel_size=(1, 1), padding='same',    activation='relu')(x)
    path2 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(filters_3x3, (1, 1), padding='same', activation='relu')(path2)
    path3 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    path3 = layers.Conv2D(filters_5x5, (1, 1), padding='same', activation='relu')(path3)
    path4 = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = layers.Conv2D(filters_pool, (1, 1), padding='same', activation='relu')(path4)
    return layers.Concatenate(axis=3)([path1, path2, path3, path4])
inp = layers.Input(shape=(224, 224, 3))

x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inp)
x = layers.MaxPooling2D(3, strides=2)(x)
x = layers.Conv2D(64, 1, strides=1, padding='same', activation='relu')(x)
x = layers.Conv2D(192, 3, strides=1, padding='same', activation='relu')(x)
x = layers.MaxPooling2D(3, strides=2)(x)
x = inception(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool=32)
x = inception(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool=64)
x = layers.MaxPooling2D(3, strides=2)(x)
x = inception(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool=64)

aux1 = layers.AveragePooling2D((5, 5), strides=3)(x)
aux1 =layers.Conv2D(128, 1, padding='same', activation='relu')(aux1)
aux1 = layers.Flatten()(aux1)
aux1 = layers.Dense(1024, activation='relu')(aux1)
aux1 = layers.Dropout(0.7)(aux1)
aux1 = layers.Dense(10, activation='softmax')(aux1)

x = inception(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24, filters_5x5=64, filters_pool=64)
x = inception(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24, filters_5x5=64, filters_pool=64)
x = inception(x, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32, filters_5x5=64, filters_pool=64)

aux2 = layers.AveragePooling2D((5, 5), strides=3)(x)
aux2 =layers.Conv2D(128, 1, padding='same', activation='relu')(aux2)
aux2 = layers.Flatten()(aux2)
aux2 = layers.Dense(1024, activation='relu')(aux2)
aux2 = layers.Dropout(0.7)(aux2)
aux2 = layers.Dense(10, activation='softmax')(aux2)

x = inception(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool=128)
x = layers.MaxPooling2D(3, strides=2)(x)
x = inception(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool=128)
x = inception(x, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5_reduce=48, filters_5x5=128, filters_pool=128)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)

out = layers.Dense(4, activation='softmax')(x)
model = Model(inputs = inp, outputs = [out, aux1, aux2])
print(model.summary())
#print("input model ", model.input)

print("output model ", model.output)

