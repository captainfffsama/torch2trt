# -*- coding: utf-8 -*-

# @Description: 用于转换 torch 中的PixelShuffle操作
# @Author: CaptainHu
# @Date: 2021-03-31 11:30:00
# @LastEditors: CaptainHu

from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.nn.functional.pixel_shuffle')
def convert_functional_pixel_shuffle(ctx):
    ctx.method_args = (torch.nn.PixelShuffle(),)+ ctx.method_args
    convert_PixelShuffle(ctx)

@tensorrt_converter('torch.nn.PixelShuffle.forward')
def convert_PixelShuffle(ctx):
    module = ctx.method_args[0]
    input=ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    upscale_factor=module.upscale_factor
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims= (int(input.shape[1]/(upscale_factor*upscale_factor)),upscale_factor,upscale_factor,input.shape[2],input.shape[3])
    layer.second_transpose= (0,3,1,4,2)

    layer = ctx.network.add_shuffle(layer.get_input(0))
    layer.reshape_dims= (int(input.shape[1]/(upscale_factor*upscale_factor)),upscale_factor*input.shape[2],upscale_factor*input.shape[3])

    output._trt = layer.get_output(0)
    