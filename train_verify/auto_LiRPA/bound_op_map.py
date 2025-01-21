from .bound_ops import *

bound_op_map = {
    'onnx::Gemm': BoundLinear,
    'prim::Constant': BoundPrimConstant,
    'grad::Concat': BoundConcatGrad,
    'grad::Relu': BoundReluGrad,
    'grad::Conv2d': BoundConv2dGrad,
    'grad::Slice': BoundSliceGrad,
    'grad::Sqr': BoundSqr,
    'grad::jacobian': BoundJacobianOP,
    'custom::Gelu': BoundGelu,
}

def register_custom_op(op_name: str, bound_obj: Bound) -> None:
    bound_op_map[op_name] = bound_obj

def unregister_custom_op(op_name: str) -> None:
    bound_op_map.pop(op_name)
