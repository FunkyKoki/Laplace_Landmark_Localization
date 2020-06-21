import os
import glob

import cv2
import torch
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from collections import OrderedDict
import matplotlib.pyplot as plt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


def getWeights(path):
    state_dict = torch.load(path, map_location='cpu')
    state_dict_use = {}
    for k, v in state_dict.items():
        state_dict_use[k] = v
    return state_dict_use

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (3, 80, 80)
    OUTPUT_NAME = "heatmap"
    OUTPUT_SIZE = (68, 80, 80)
    DTYPE = trt.float32

def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # blockE1
    blockE1_0_conv_w = weights['blockE1.0.weight'].numpy()
    blockE1_0_conv_b = weights['blockE1.0.bias'].numpy()
    blockE1_0_conv = network.add_convolution(input=input_tensor, num_output_maps=64, kernel_shape=(3, 3), kernel=blockE1_0_conv_w, bias=blockE1_0_conv_b)
    blockE1_0_conv.stride = (1, 1)
    blockE1_0_conv.padding = (1, 1)

    blockE1_lrelu = network.add_activation(input=blockE1_0_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockE1_lrelu.alpha = 0.2

    # blockE2
    blockE2_pool = network.add_pooling(input=blockE1_lrelu.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    blockE2_pool.stride = (2, 2)

    blockE2_1_conv_w = weights['blockE2.1.weight'].numpy()
    blockE2_1_conv_b = weights['blockE2.1.bias'].numpy()
    blockE2_1_conv = network.add_convolution(blockE2_pool.get_output(0), 64, (3, 3), blockE2_1_conv_w, blockE2_1_conv_b)
    blockE2_1_conv.stride = (1, 1)
    blockE2_1_conv.padding = (1, 1)

    blockE2_lrelu = network.add_activation(input=blockE2_1_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockE2_lrelu.alpha = 0.2

    # blockE3
    blockE3_pool = network.add_pooling(input=blockE2_lrelu.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    blockE3_pool.stride = (2, 2)

    blockE3_1_conv_w = weights['blockE3.1.weight'].numpy()
    blockE3_1_conv_b = weights['blockE3.1.bias'].numpy()
    blockE3_1_conv = network.add_convolution(blockE3_pool.get_output(0), 64, (3, 3), blockE3_1_conv_w, blockE3_1_conv_b)
    blockE3_1_conv.stride = (1, 1)
    blockE3_1_conv.padding = (1, 1)

    blockE3_lrelu = network.add_activation(input=blockE3_1_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockE3_lrelu.alpha = 0.2

    # blockE4
    blockE4_pool = network.add_pooling(input=blockE3_lrelu.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    blockE4_pool.stride = (2, 2)

    blockE4_1_conv_w = weights['blockE4.1.weight'].numpy()
    blockE4_1_conv_b = weights['blockE4.1.bias'].numpy()
    blockE4_1_conv = network.add_convolution(blockE4_pool.get_output(0), 64, (3, 3), blockE4_1_conv_w, blockE4_1_conv_b)
    blockE4_1_conv.stride = (1, 1)
    blockE4_1_conv.padding = (1, 1)

    blockE4_lrelu = network.add_activation(input=blockE4_1_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockE4_lrelu.alpha = 0.2

    # blockD4
    blockD4_pool = network.add_pooling(input=blockE4_lrelu.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    blockD4_pool.stride = (2, 2)

    blockD4_1_conv_w = weights['blockD4.1.weight'].numpy()
    blockD4_1_conv_b = weights['blockD4.1.bias'].numpy()
    blockD4_1_conv = network.add_convolution(blockD4_pool.get_output(0), 64, (3, 3), blockD4_1_conv_w, blockD4_1_conv_b)
    blockD4_1_conv.stride = (1, 1)
    blockD4_1_conv.padding = (1, 1)

    blockD4_lrelu_1 = network.add_activation(input=blockD4_1_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockD4_lrelu_1.alpha = 0.2

    blockD4_4_conv_w = weights['blockD4.4.weight'].numpy()
    blockD4_4_conv_b = weights['blockD4.4.bias'].numpy()
    blockD4_4_conv = network.add_convolution(blockD4_lrelu_1.get_output(0), 64, (1, 1), blockD4_4_conv_w, blockD4_4_conv_b)
    blockD4_4_conv.stride = (1, 1)

    blockD4_lrelu_2 = network.add_activation(input=blockD4_4_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockD4_lrelu_2.alpha = 0.2

    blockD4_upsample = network.add_resize(input=blockD4_lrelu_2.get_output(0))
    blockD4_upsample.shape = [64, 10, 10]

    # concat
    concat_1 = network.add_concatenation(inputs=[blockD4_upsample.get_output(0), blockE4_lrelu.get_output(0)])
    concat_1.axis = 0
    
    # blockD3
    blockD3_0_conv_w = weights['blockD3.0.weight'].numpy()
    blockD3_0_conv_b = weights['blockD3.0.bias'].numpy()
    blockD3_0_conv = network.add_convolution(concat_1.get_output(0), 64, (3, 3), blockD3_0_conv_w, blockD3_0_conv_b)
    blockD3_0_conv.stride = (1, 1)
    blockD3_0_conv.padding = (1, 1)

    blockD3_lrelu_1 = network.add_activation(input=blockD3_0_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockD3_lrelu_1.alpha = 0.2

    blockD3_3_conv_w = weights['blockD3.3.weight'].numpy()
    blockD3_3_conv_b = weights['blockD3.3.bias'].numpy()
    blockD3_3_conv = network.add_convolution(blockD3_lrelu_1.get_output(0), 64, (1, 1), blockD3_3_conv_w, blockD3_3_conv_b)
    blockD3_3_conv.stride = (1, 1)

    blockD3_lrelu_2 = network.add_activation(input=blockD3_3_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockD3_lrelu_2.alpha = 0.2

    blockD3_upsample = network.add_resize(input=blockD3_lrelu_2.get_output(0))
    blockD3_upsample.shape = [64, 20, 20]

    # concat
    concat_2 = network.add_concatenation(inputs=[blockD3_upsample.get_output(0), blockE3_lrelu.get_output(0)])
    concat_2.axis = 0

    # blockD2
    blockD2_0_conv_w = weights['blockD2.0.weight'].numpy()
    blockD2_0_conv_b = weights['blockD2.0.bias'].numpy()
    blockD2_0_conv = network.add_convolution(concat_2.get_output(0), 64, (3, 3), blockD2_0_conv_w, blockD2_0_conv_b)
    blockD2_0_conv.stride = (1, 1)
    blockD2_0_conv.padding = (1, 1)

    blockD2_lrelu_1 = network.add_activation(input=blockD2_0_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockD2_lrelu_1.alpha = 0.2

    blockD2_3_conv_w = weights['blockD2.3.weight'].numpy()
    blockD2_3_conv_b = weights['blockD2.3.bias'].numpy()
    blockD2_3_conv = network.add_convolution(blockD2_lrelu_1.get_output(0), 64, (1, 1), blockD2_3_conv_w, blockD2_3_conv_b)
    blockD2_3_conv.stride = (1, 1)

    blockD2_lrelu_2 = network.add_activation(input=blockD2_3_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockD2_lrelu_2.alpha = 0.2

    blockD2_upsample = network.add_resize(input=blockD2_lrelu_2.get_output(0))
    blockD2_upsample.shape = [64, 40, 40]

    # concat
    concat_3 = network.add_concatenation(inputs=[blockD2_upsample.get_output(0), blockE2_lrelu.get_output(0)])
    concat_3.axis = 0

    # blockD1
    blockD1_0_conv_w = weights['blockD1.0.weight'].numpy()
    blockD1_0_conv_b = weights['blockD1.0.bias'].numpy()
    blockD1_0_conv = network.add_convolution(concat_3.get_output(0), 64, (3, 3), blockD1_0_conv_w, blockD1_0_conv_b)
    blockD1_0_conv.stride = (1, 1)
    blockD1_0_conv.padding = (1, 1)

    blockD1_lrelu_1 = network.add_activation(input=blockD1_0_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockD1_lrelu_1.alpha = 0.2

    blockD1_3_conv_w = weights['blockD1.3.weight'].numpy()
    blockD1_3_conv_b = weights['blockD1.3.bias'].numpy()
    blockD1_3_conv = network.add_convolution(blockD1_lrelu_1.get_output(0), 64, (1, 1), blockD1_3_conv_w, blockD1_3_conv_b)
    blockD1_3_conv.stride = (1, 1)

    blockD1_lrelu_2 = network.add_activation(input=blockD1_3_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockD1_lrelu_2.alpha = 0.2

    blockD1_upsample = network.add_resize(input=blockD1_lrelu_2.get_output(0))
    blockD1_upsample.shape = [64, 80, 80]

    # concat
    concat_4 = network.add_concatenation(inputs=[blockD1_upsample.get_output(0), blockE1_lrelu.get_output(0)])
    concat_4.axis = 0

    # blockT
    blockT_0_conv_w = weights['blockT.0.weight'].numpy()
    blockT_0_conv_b = weights['blockT.0.bias'].numpy()
    blockT_0_conv = network.add_convolution(concat_4.get_output(0), 128, (5, 5), blockT_0_conv_w, blockT_0_conv_b)
    blockT_0_conv.stride = (1, 1)
    blockT_0_conv.padding = (2, 2)

    blockT_lrelu_1 = network.add_activation(input=blockT_0_conv.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockT_lrelu_1.alpha = 0.2

    blockT_3_conv_w = weights['blockT.3.weight'].numpy()
    blockT_3_conv_b = weights['blockT.3.bias'].numpy()
    blockT_3_conv = network.add_convolution(blockT_lrelu_1.get_output(0), 68, (1, 1), blockT_3_conv_w, blockT_3_conv_b)
    blockT_3_conv.stride = (1, 1)

    blockT_4_bn_w = weights['blockT.4.weight'].numpy()
    blockT_4_bn_b = weights['blockT.4.bias'].numpy()
    blockT_4_bn_m = weights['blockT.4.running_mean'].numpy()
    blockT_4_bn_v = weights['blockT.4.running_var'].numpy()
    blockT_4_bn_scale = blockT_4_bn_w / np.sqrt(blockT_4_bn_v + 1e-5)
    blockT_4_bn_shift = -blockT_4_bn_m / np.sqrt(blockT_4_bn_v + 1e-5) * blockT_4_bn_w + blockT_4_bn_b
    blockT_4_bn_power = np.ones(len(blockT_4_bn_w), dtype=np.float32)
    blockT_4_bn = network.add_scale(blockT_3_conv.get_output(0), trt.ScaleMode.CHANNEL, blockT_4_bn_shift, blockT_4_bn_scale, blockT_4_bn_power)

    blockT_lrelu_2 = network.add_activation(input=blockT_4_bn.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockT_lrelu_2.alpha = 0.2

    blockT_7_conv_w = weights['blockT.7.weight'].numpy()
    blockT_7_conv_b = weights['blockT.7.bias'].numpy()
    blockT_7_conv = network.add_convolution(blockT_lrelu_2.get_output(0), 68, (1, 1), blockT_7_conv_w, blockT_7_conv_b)
    blockT_7_conv.stride = (1, 1)

    blockT_8_bn_w = weights['blockT.8.weight'].numpy()
    blockT_8_bn_b = weights['blockT.8.bias'].numpy()
    blockT_8_bn_m = weights['blockT.8.running_mean'].numpy()
    blockT_8_bn_v = weights['blockT.8.running_var'].numpy()
    blockT_8_bn_scale = blockT_8_bn_w / np.sqrt(blockT_8_bn_v + 1e-5)
    blockT_8_bn_shift = -blockT_8_bn_m / np.sqrt(blockT_8_bn_v + 1e-5) * blockT_8_bn_w + blockT_8_bn_b
    blockT_8_bn_power = np.ones(len(blockT_8_bn_w), dtype=np.float32)
    blockT_8_bn = network.add_scale(blockT_7_conv.get_output(0), trt.ScaleMode.CHANNEL, blockT_8_bn_shift, blockT_8_bn_scale, blockT_8_bn_power)

    blockT_lrelu_3 = network.add_activation(input=blockT_8_bn.get_output(0), type=trt.ActivationType.LEAKY_RELU)
    blockT_lrelu_3.alpha = 0.2

    blockT_10_conv_w = weights['blockT.10.weight'].numpy()
    blockT_10_conv_b = weights['blockT.10.bias'].numpy()
    blockT_10_conv = network.add_convolution(blockT_lrelu_3.get_output(0), 68, (1, 1), blockT_10_conv_w, blockT_10_conv_b)
    blockT_10_conv.stride = (1, 1)

    blockT_10_conv.get_output(0).name = ModelData.OUTPUT_NAME

    network.mark_output(tensor=blockT_10_conv.get_output(0))


def build_engine(weights, engine_file_path):
    if os.path.exists(engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = 2 << 30  # 4 GiB
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # Build and return an engine.
        engine = builder.build_cuda_engine(network)
        assert engine is not None
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine


# build_engine(getWeights('./20200515_face300W80_Train_2_model_best.pth'), './20200515_face300W80_Train_2_model_best.trt')


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def getImg(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    mean, std = cv2.meanStdDev(img/255.)
    for ch in range(3):
        if std[ch, 0] < 1e-6:
            std[ch, 0] = 1.
    mean = np.array([[[mean[0, 0], mean[1, 0], mean[2, 0]]]])
    std = np.array([[[std[0, 0], std[1, 0], std[2, 0]]]])
    img = (img / 255. - mean) / std
    return img.transpose((2, 0, 1))


def showImg(pic, name):
    plt.figure(name)
    plt.imshow(pic)
    plt.axis('off')
    plt.title(name)
    plt.show()


import time
def main():
    with build_engine(getWeights('./20200523_face300W80_Train_5_model_best.pth'), './20200523_face300W80_Train_5_model_best.trt') as engine:
        # Build an engine, allocate buffers and create a stream.
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            img_paths = glob.glob('/home/jin/Desktop/common/*')
            startTime = time.time()
            for img_path in img_paths:
                img = getImg(img_path)
                np.copyto(inputs[0].host, img.flatten().astype(np.float32))
                [output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                output = np.resize(output, (68, 80, 80))
            print(time.time()-startTime)

    # build_engine(getWeights('./20200515_face300W80_Train_2_model_best.pth'), './20200515_face300W80_Train_2_model_best.trt')
            
main()