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
class PatchedCudaEngine:
    def __init__(self, engine):
        self._engine = engine
        
    def __getattr__(self, name):
        # Primero buscar en el engine original
        return getattr(self._engine, name)
    
    def __getitem__(self, key):
        """Hacer el objeto subscriptable"""
        return self._engine[key]
    
    def __setitem__(self, key, value):
        """Permitir asignación por índice"""
        self._engine[key] = value
    
    @property
    def num_bindings(self):
        """Propiedad para compatibilidad"""
        if hasattr(self._engine, 'num_bindings'):
            return self._engine.num_bindings
        elif hasattr(self._engine, 'num_io_tensors'):
            return self._engine.num_io_tensors
        else:
            return 0
    
    @property
    def num_optimization_profiles(self):
        """Propiedad para compatibilidad"""
        if hasattr(self._engine, 'num_optimization_profiles'):
            return self._engine.num_optimization_profiles
        else:
            return 1
    
    def get_binding_dtype(self, index):
        """Compatibilidad para TensorRT 10.x"""
        if hasattr(self._engine, 'get_binding_dtype'):
            return self._engine.get_binding_dtype(index)
        elif hasattr(self._engine, 'get_tensor_dtype'):
            # TensorRT 10.x usa get_tensor_dtype con nombres
            if hasattr(self._engine, 'get_tensor_name'):
                tensor_name = self._engine.get_tensor_name(index)
            elif hasattr(self._engine, 'get_binding_name'):
                tensor_name = self._engine.get_binding_name(index)
            else:
                tensor_name = f"binding_{index}"
            return self._engine.get_tensor_dtype(tensor_name)
        else:
            # Fallback - asumir float16
            return trt.DataType.HALF
    
    def get_binding_shape(self, index):
        """Compatibilidad para TensorRT 10.x"""
        if hasattr(self._engine, 'get_binding_shape'):
            return self._engine.get_binding_shape(index)
        elif hasattr(self._engine, 'get_tensor_shape'):
            # TensorRT 10.x usa get_tensor_shape con nombres
            if hasattr(self._engine, 'get_tensor_name'):
                tensor_name = self._engine.get_tensor_name(index)
            elif hasattr(self._engine, 'get_binding_name'):
                tensor_name = self._engine.get_binding_name(index)
            else:
                tensor_name = f"binding_{index}"
            return self._engine.get_tensor_shape(tensor_name)
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
            # TensorRT 10.x
            if hasattr(self._engine, 'get_tensor_name'):
                tensor_name = self._engine.get_tensor_name(index)
            elif hasattr(self._engine, 'get_binding_name'):
                tensor_name = self._engine.get_binding_name(index)
            else:
                tensor_name = f"binding_{index}"
            mode = self._engine.get_tensor_mode(tensor_name)
            return mode == trt.TensorIOMode.INPUT
        else:
            # Fallback
            return index == 0

# Monkey patch para interceptar la creación de engines
original_deserialize = trt.Runtime.deserialize_cuda_engine

def patched_deserialize(self, *args, **kwargs):
    engine = original_deserialize(self, *args, **kwargs)
    if engine:
        return PatchedCudaEngine(engine)
    return engine

trt.Runtime.deserialize_cuda_engine = patched_deserialize
