# randn_patch.py  v3 — usa diffusers.utils.randn_tensor
import torch
from functools import wraps
from diffusers.utils import randn_tensor

_orig_randn = torch.randn  # copia de la original (por si hace falta)

@wraps(_orig_randn)
def _randn_patched(size, *args, generator=None, **kwargs):
    """
    Parche definitivo:
    • Siempre llama a diffusers.utils.randn_tensor, que maneja TODOS los casos.
    • generator:
        - None  -> seed global, 100 % compatible
        - torch.Generator -> usa ese generador
        - lista/tupla de generadores -> batch con seeds distintos
    • dtype, device se respetan; resto de kwargs se ignoran (no los usa randn_tensor).
    """
    dtype  = kwargs.get("dtype", None)
    device = kwargs.get("device", None)

    # Caso lista/tupla -> generar un tensor por seed y concatenar
    if isinstance(generator, (list, tuple)):
        tensors = [
            randn_tensor(size, generator=g, device=device, dtype=dtype)
            for g in generator
        ]
        return torch.stack(tensors, dim=0)

    # Caso normal (None o torch.Generator)
    return randn_tensor(size, generator=generator, device=device, dtype=dtype)

# Activar parche
torch.randn = _randn_patched
print("✅ Parche randn() v3 (randn_tensor) activado")
