import os
import copy

import numpy as np

import cv2
import dlib

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import torch

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


def use_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        pass


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

rescale_ratio = 0.45
temperature = 2.0

spatialRange = np.array(range(80))+1

with use_engine('./20200523_face300W80_Train_5_model_best.trt') as engine:
    # Build an engine, allocate buffers and create a stream.
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    with engine.create_execution_context() as context:
        count = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        write = cv2.VideoWriter('./xiao2Det.avi',fourcc, 25.0, (640,360))

        cameraCapture = cv2.VideoCapture('./xiao2.mp4')
        print(cameraCapture.get(cv2.CAP_PROP_FPS), cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT), cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        success, frame = cameraCapture.read()
        detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

        while success:
            count += 1
            if count > 2400:
                break
            success, frame = cameraCapture.read()
            faces = detector(frame, 0)

            for _ in range(len(faces)):

                d = faces.pop().rect

                height = d.bottom() - d.top()
                width = d.right() - d.left()
                bbox = [
                    int(d.left() - rescale_ratio * width),
                    int(d.top() - rescale_ratio * height),
                    int(d.right() + rescale_ratio * width),
                    int(d.bottom() + rescale_ratio * height)
                ]
                position_before = np.float32([
                    [int(bbox[0]), int(bbox[1])],
                    [int(bbox[0]), int(bbox[3])],
                    [int(bbox[2]), int(bbox[3])]
                ])
                position_after = np.float32([
                    [0, 0],
                    [0, 79],
                    [79, 79]
                ])
                crop_matrix = cv2.getAffineTransform(position_before, position_after)
                img = cv2.warpAffine(frame, crop_matrix, (80, 80))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mean, std = cv2.meanStdDev(img/255.)
                for ch in range(3):
                    if std[ch, 0] < 1e-6:
                        std[ch, 0] = 1.
                mean = np.array([[[mean[0, 0], mean[1, 0], mean[2, 0]]]])
                std = np.array([[[std[0, 0], std[1, 0], std[2, 0]]]])
                img = (img / 255. - mean) / std
                img = img.transpose((2, 0, 1))

                np.copyto(inputs[0].host, img.flatten().astype(np.float32))
                [output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                output = np.resize(output, (68, 80, 80))

                hDistributionTemp = temperature*np.sum(output, axis=2)
                hDistribution = np.exp(hDistributionTemp)/np.sum(np.exp(hDistributionTemp), axis=1, keepdims=True)
                wDistributionTemp = temperature*np.sum(output, axis=1)
                wDistribution = np.exp(wDistributionTemp)/np.sum(np.exp(wDistributionTemp), axis=1, keepdims=True)
                
                hMean = np.sum(hDistribution*spatialRange, axis=1)
                wMean = np.sum(wDistribution*spatialRange, axis=1)

                output = np.stack((wMean, hMean), axis=1)

                for i in range(68):
                    output[i, 0] = bbox[0] + output[i, 0] / 80 * (bbox[2] - bbox[0])
                    output[i, 1] = bbox[1] + output[i, 1] / 80 * (bbox[3] - bbox[1])
                    cv2.circle(frame, (int(output[i, 0]), int(output[i, 1])), 2, (0,0,255), -1)
                
                cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 2)

            cv2.imshow("Camera", frame)
            write.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                torch.cuda.empty_cache()
                break

cameraCapture.release()
write.release()
cv2.destroyAllWindows()