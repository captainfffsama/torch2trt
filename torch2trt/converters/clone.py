# -*- coding: utf-8 -*-

# @Description:  用于辅助 Tensor.clone 的转换
# @Author: CaptainHu
# @Date: 2021-03-31 17:06:27
# @LastEditors: CaptainHu


from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test,add_single_module_test

# FIXME: 过测试提示:[TensorRT] ERROR: INVALID_ARGUMENT: Cannot find binding of given name: input_0
# @tensorrt_converter('torch.clone')
# @tensorrt_converter('torch.Tensor.clone')
def convert_Tensor_clone(ctx):
    input=ctx.method_args[0]
    output=ctx.method_return

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    dtype = check_torch_dtype(input)
    scalar = input.detach().cpu().numpy()
    output._trt= ctx.network.add_constant(input_trt.shape, scalar).get_output(0)
    

class TensorClone(torch.nn.Module):
    def __init__(self):
        super(TensorClone, self).__init__()

    def forward(self,x):
        return x.clone()

# @add_single_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
# @add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_Tensor_clone():
    return TensorClone()