import argparse
import copy
import os
import os.path as osp
import time
import warnings
import typing
import torch
from numbers import Number
from typing import Any, Callable, List, Optional, Union
from numpy import prod
import numpy as np
from collections import Counter
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import batchnorm_flop_jit, matmul_flop_jit, generic_activation_jit, get_shape
import psutil
def softmax_jit(inputs: typing.List[object], outputs: typing.List[object]) -> typing.Counter[str]:
    input_shape = get_shape(inputs[0])
    output_shape = get_shape(outputs[0])
    flop = prod(input_shape) * 2 + prod(output_shape) # exponentiating & summing inputs + denominating in each batch
    flop_counter = Counter({"softmax": flop})
    return flop_counter


def simple_op_flop_jit(inputs: List[Any], outputs: List[Any], op_per_element: int) -> Number:
    """
    Generic handler for simple operations like addition and multiplication.
    op_per_element: Number of FLOPs per element (1 for add or mul)
    """
    input_shape = inputs[0].type().sizes()
    num_elements = prod(input_shape)
    return num_elements * op_per_element

def add_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    return simple_op_flop_jit(inputs, outputs, 1)

def mul_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    return simple_op_flop_jit(inputs, outputs, 1)




def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the rfft/rfftn operator.
    """
    input_shape = inputs[0].type().sizes()
    B, H, W, C = input_shape
    N = H * W
    flops = N * C * np.ceil(np.log2(N))
    return flops

def calc_flops(model, img_w=256, img_h = 256, show_details=False, ratios=None):
    with torch.no_grad():
        x = torch.randn(1, 3, img_w, img_h) # 256x256 for mpii, 192,256 for coco 
        model.default_ratio = ratios
        fca1 = FlopCountAnalysis(model, x)
        handlers = {
            "aten::batch_norm": batchnorm_flop_jit,
            "aten::group_norm": batchnorm_flop_jit,
            "aten::layer_norm": batchnorm_flop_jit,
            'aten::fft_rfft2': rfft_flop_jit,
            'aten::fft_irfft2': rfft_flop_jit,
            "aten::add": generic_activation_jit("add"),
            "aten::sub": generic_activation_jit("sub"),
            "aten::mul": generic_activation_jit("mul"),
            "aten::div": generic_activation_jit("div"),
            "aten::sqrt": generic_activation_jit("sqrt"),
            "aten::sigmoid": generic_activation_jit("sigmoid"),
            "aten::sigmoid_": generic_activation_jit("sigmoid_"),
            "aten::relu": generic_activation_jit("relu"),
            "aten::relu_": generic_activation_jit("relu_"),
            "aten::gelu": generic_activation_jit("gelu"),
            "aten::add_": generic_activation_jit("add_"),
            "aten::sub_": generic_activation_jit("sub_"),
            "aten::mul_": generic_activation_jit("mul_"),
            "aten::div_": generic_activation_jit("div_"),
            "aten::sqrt_": generic_activation_jit("sqrt_"),
        }
        fca1.set_op_handle(**handlers)
        flops1 = fca1.total()
        if show_details:
            print(fca1.by_module())
        print("#### GFLOPs: {} for ratio {}".format(flops1 / 1e9, ratios))
    return flops1 / 1e9

@torch.no_grad()
def throughput(images, model):
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)
    batch_size = images.shape[0]
    for i in range(50):
        model(images)
    torch.cuda.synchronize()
    print(f"throughput averaged with 30 times")
    tic1 = time.time()
    for i in range(30):
        model(images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
    MB = 1024.0 * 1024.0
    #print('memory:', torch.cuda.max_memory_allocated(device=device) / MB)
    process = psutil.Process(os.getpid())
    print('Memory:', process.memory_info().rss / MB, 'MB')

    return f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}",f'memory: {process.memory_info().rss / MB}'