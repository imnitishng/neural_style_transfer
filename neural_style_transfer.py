#%% Loading images

from keras.preprocessing.image import load_img, img_to_array

target_image_path = 'C:/Users/Nitish/Downloads/pp.jfif'
style_reference_image_path = 'C:/Users/Nitish/Downloads/wiggly2.jpeg'

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

#%% Auxillary Functions

import numpy as np
from keras.applications import vgg19

def preprocess_img(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_img(x):
    # preprocess of vgg19 are reverted back to normal
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    #BGR to RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
    
#%% Pretrained VGG19 and apply to 3 images

from keras import backend as K

# the 2 will remain constant
target_image  = K.constant(preprocess_img(target_image_path))
style_reference_image = K.constant(preprocess_img(style_reference_image_path))

# this will change over time so placeholder
combination_image = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([target_image, 
                              style_reference_image,
                              combination_image], axis=0)
    
model = vgg19.VGG19(input_tensor=input_tensor, 
                    weights='imagenet', 
                    include_top=False)

print('Model Loaded')

#%% Losses

# Content Loss
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# Style Loss
def gram_matrix(x):
    # permutedimension changes channel last to first
    # batch flatten gets (64, 400, 400) to (64, 160000)
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    
    # transpose gets (64, 160000) to (160000, 64)
    # filters are multiplied (64, 64)
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

#%% Total Variation Loss
    
def total_variation_loss(x):
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] - 
                 x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] - 
                 x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

#%% loss to minimize
    
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layers = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

loss = K.variable(0.)
layer_features = outputs_dict[content_layers]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, : ,: ,:]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl

loss += total_variation_weight * total_variation_loss(combination_image)

#%% Gradient Descent Process

grads = K.gradients(loss, combination_image)[0]

# actually target image is passed in the below line maybe 
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

class Evaluator(object):
    
    def __init__(self):
        self.loss_value = None
        self.grad_values = None
    
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

#%% Style-transfer Loop

from scipy.optimize import fmin_l_bfgs_b
import time
import imageio

result_prefix = 'pp_wiggly'
iterations = 20
save_path = 'C:/Users/Nitish/Keras progs/Neural Style Transfer OPS'

x = preprocess_img(target_image_path)
x = x.flatten()
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
    print('Current Loss value : ', min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_img(img)
    fname = save_path + '/' + result_prefix + '_at_iteration_%d.png' % i
    imageio.imwrite(fname, img)
    print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' %(i, end_time - start_time))
    






