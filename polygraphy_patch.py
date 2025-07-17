# polygraphy_patch.py
import polygraphy.backend.trt.util as trt_util
import tensorrt as trt

def get_bindings_per_profile(engine):
    """
    Función de compatibilidad para versiones nuevas de polygraphy
    """
    if hasattr(engine, 'num_io_tensors'):
        # TensorRT 10.x
        return engine.num_io_tensors // engine.num_optimization_profiles
    elif hasattr(engine, 'num_bindings'):
        # TensorRT 8.x/9.x
        return engine.num_bindings // engine.num_optimization_profiles
    else:
        # Fallback
        return 1

# Agregar la función al módulo
trt_util.get_bindings_per_profile = get_bindings_per_profile

# Parche para métodos deprecados de ICudaEngine
original_cuda_engine = trt.ICudaEngine

class PatchedCudaEngine:
    def __init__(self, engine):
        self._engine = engine
        # Copiar todos los atributos del engine original
        for attr in dir(engine):
            if not attr.startswith('_') and not hasattr(self, attr):
                try:
                    setattr(self, attr, getattr(engine, attr))
                except:
                    pass
    
    def __getattr__(self, name):
        # Si el atributo no existe, intentar obtenerlo del engine original
        return getattr(self._engine, name)
    
    def get_binding_dtype(self, index):
        """Compatibilidad para TensorRT 10.x"""
        if hasattr(self._engine, 'get_binding_dtype'):
            return self._engine.get_binding_dtype(index)
        elif hasattr(self._engine, 'get_tensor_dtype'):
            # TensorRT 10.x usa get_tensor_dtype
            binding_name = self.get_tensor_name(index) if hasattr(self, 'get_tensor_name') else self.get_binding_name(index)
            return self._engine.get_tensor_dtype(binding_name)
        else:
            # Fallback - asumir float16
            return trt.DataType.HALF
    
    def get_binding_shape(self, index):
        """Compatibilidad para TensorRT 10.x"""
        if hasattr(self._engine, 'get_binding_shape'):
            return self._engine.get_binding_shape(index)
        elif hasattr(self._engine, 'get_tensor_shape'):
            # TensorRT 10.x usa get_tensor_shape
            binding_name = self.get_tensor_name(index) if hasattr(self, 'get_tensor_name') else self.get_binding_name(index)
            return self._engine.get_tensor_shape(binding_name)
        else:
            return []
    
    def get_binding_name(self, index):
        """Compatibilidad para TensorRT 10.x"""
        if hasattr(self._engine, 'get_binding_name'):
            return self._engine.get_binding_name(index)
        elif hasattr(self._engine, 'get_tensor_name'):
            return self._engine.get_tensor_name(index)
        else:
            return f"binding_{index}"
    
    def binding_is_input(self, index):
        """Compatibilidad para TensorRT 10.x"""
        if hasattr(self._engine, 'binding_is_input'):
            return self._engine.binding_is_input(index)
        elif hasattr(self._engine, 'get_tensor_mode'):
            binding_name = self.get_tensor_name(index) if hasattr(self, 'get_tensor_name') else self.get_binding_name(index)
            return self._engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT
        else:
            return index == 0  # Asumir que el primer binding es input

# Monkey patch para interceptar la creación de engines
original_deserialize = trt.Runtime.deserialize_cuda_engine

def patched_deserialize(self, *args, **kwargs):
    engine = original_deserialize(self, *args, **kwargs)
    if engine:
        return PatchedCudaEngine(engine)
    return engine

trt.Runtime.deserialize_cuda_engine = patched_deserialize
