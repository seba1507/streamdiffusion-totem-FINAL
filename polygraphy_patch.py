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

# Clase wrapper que delega todo al engine original
class PatchedCudaEngine:
    def __init__(self, engine):
        self._engine = engine
        
    def __getattr__(self, name):
        # Si es un método de compatibilidad, usarlo
        if name in ['get_binding_dtype', 'get_binding_shape', 'get_binding_name', 'binding_is_input']:
            return getattr(self, f'_compat_{name}')
        # Si no, delegar al engine original
        return getattr(self._engine, name)
    
    def __getitem__(self, key):
        """Hacer el objeto subscriptable"""
        return self._engine.__getitem__(key)
    
    def __setitem__(self, key, value):
        """Permitir asignación por índice"""
        return self._engine.__setitem__(key, value)
    
    def __len__(self):
        """Soporte para len()"""
        return self._engine.__len__()
    
    def __repr__(self):
        """Representación del objeto"""
        return f"PatchedCudaEngine({self._engine})"
    
    # Métodos de compatibilidad
    def _compat_get_binding_dtype(self, index):
        """Compatibilidad para TensorRT 10.x"""
        try:
            return self._engine.get_binding_dtype(index)
        except AttributeError:
            # TensorRT 10.x
            try:
                tensor_name = self._engine.get_tensor_name(index)
                return self._engine.get_tensor_dtype(tensor_name)
            except:
                return trt.DataType.HALF
    
    def _compat_get_binding_shape(self, index):
        """Compatibilidad para TensorRT 10.x"""
        try:
            return self._engine.get_binding_shape(index)
        except AttributeError:
            # TensorRT 10.x
            try:
                tensor_name = self._engine.get_tensor_name(index)
                return self._engine.get_tensor_shape(tensor_name)
            except:
                return []
    
    def _compat_get_binding_name(self, index):
        """Compatibilidad para TensorRT 10.x"""
        try:
            return self._engine.get_binding_name(index)
        except AttributeError:
            # TensorRT 10.x
            try:
                return self._engine.get_tensor_name(index)
            except:
                return f"binding_{index}"
    
    def _compat_binding_is_input(self, index):
        """Compatibilidad para TensorRT 10.x"""
        try:
            return self._engine.binding_is_input(index)
        except AttributeError:
            # TensorRT 10.x
            try:
                tensor_name = self._engine.get_tensor_name(index)
                mode = self._engine.get_tensor_mode(tensor_name)
                return mode == trt.TensorIOMode.INPUT
            except:
                return index == 0

# Alternativa: parchear directamente los métodos en el módulo runtime
# en lugar de wrappear el engine
import types

def patch_engine_methods():
    """Agregar métodos de compatibilidad directamente a ICudaEngine"""
    engine_class = trt.ICudaEngine
    
    # Guardar los métodos originales si existen
    original_methods = {}
    for method in ['get_binding_dtype', 'get_binding_shape', 'get_binding_name', 'binding_is_input']:
        if hasattr(engine_class, method):
            original_methods[method] = getattr(engine_class, method)
    
    def patched_get_binding_dtype(self, index):
        if 'get_binding_dtype' in original_methods:
            try:
                return original_methods['get_binding_dtype'](self, index)
            except:
                pass
        # Fallback para TensorRT 10
        try:
            tensor_name = self.get_tensor_name(index)
            return self.get_tensor_dtype(tensor_name)
        except:
            return trt.DataType.HALF
    
    def patched_get_binding_shape(self, index):
        if 'get_binding_shape' in original_methods:
            try:
                return original_methods['get_binding_shape'](self, index)
            except:
                pass
        # Fallback para TensorRT 10
        try:
            tensor_name = self.get_tensor_name(index)
            return self.get_tensor_shape(tensor_name)
        except:
            return []
    
    def patched_get_binding_name(self, index):
        if 'get_binding_name' in original_methods:
            try:
                return original_methods['get_binding_name'](self, index)
            except:
                pass
        # Fallback para TensorRT 10
        try:
            return self.get_tensor_name(index)
        except:
            return f"binding_{index}"
    
    def patched_binding_is_input(self, index):
        if 'binding_is_input' in original_methods:
            try:
                return original_methods['binding_is_input'](self, index)
            except:
                pass
        # Fallback para TensorRT 10
        try:
            tensor_name = self.get_tensor_name(index)
            mode = self.get_tensor_mode(tensor_name)
            return mode == trt.TensorIOMode.INPUT
        except:
            return index == 0
    
    # Aplicar los parches
    if not hasattr(engine_class, 'get_binding_dtype'):
        engine_class.get_binding_dtype = patched_get_binding_dtype
    if not hasattr(engine_class, 'get_binding_shape'):
        engine_class.get_binding_shape = patched_get_binding_shape
    if not hasattr(engine_class, 'get_binding_name'):
        engine_class.get_binding_name = patched_get_binding_name
    if not hasattr(engine_class, 'binding_is_input'):
        engine_class.binding_is_input = patched_binding_is_input

# Aplicar los parches directamente
patch_engine_methods()
