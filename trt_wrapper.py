import torch, tensorrt as trt

class TrtEngine:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context  = self.engine.create_execution_context()
        self.bindings = [None] * self.engine.num_bindings
        self.stream   = torch.cuda.current_stream().cuda_stream

    def __call__(self, *inputs):
        # Soporta 1 input → 1 output (suficiente para VAE y UNet)
        inp = inputs[0]
        self.bindings[0] = int(inp.data_ptr())
        if not hasattr(self, "_out"):
            out_shape = tuple(self.engine.get_binding_shape(1))
            self._out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
            self.bindings[1] = int(self._out.data_ptr())
        self.context.execute_async_v2(self.bindings, self.stream)
        return self._out
