# randn_patch.py  v5 — None seguro + listas + generador único
import torch
from functools import wraps

_orig_randn = torch.randn  # copia original

def _randn_single(shape, *, generator, device, dtype):
    """Devuelve un tensor de ruido para un único generador (o None)."""
    if generator is None:
        # Llamada sin el kwarg generator ⇒ PyTorch no se queja
        return _orig_randn(shape, device=device, dtype=dtype)
    if isinstance(generator, torch.Generator):
        return _orig_randn(shape, generator=generator, device=device, dtype=dtype)
    # Tipo raro → ignoramos generator y delegamos a PyTorch (dejará su propio error)
    return _orig_randn(shape, device=device, dtype=dtype)

@wraps(_orig_randn)
def _randn_patched(shape, *args, generator=None, **kwargs):
    """
    Soporta:
    • generator=None
    • generator=torch.Generator
    • generator=list/tuple[torch.Generator]
    """
    device = kwargs.pop("device", None)
    dtype  = kwargs.pop("dtype",  None)

    # Lista/tupla ⇒ iterar y apilar
    if isinstance(generator, (list, tuple)):
        tensors = [
            _randn_single(shape, generator=g, device=device, dtype=dtype)
            for g in generator
        ]
        return torch.stack(tensors, dim=0)

    # Caso único / None
    return _randn_single(shape, generator=generator, device=device, dtype=dtype)

torch.randn = _randn_patched
print("✅ Parche randn() v5 activado (None seguro)")
