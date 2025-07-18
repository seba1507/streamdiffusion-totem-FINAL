# randn_patch.py
import torch
from functools import wraps

_orig_randn = torch.randn  # guardamos la versión original

@wraps(_orig_randn)
def _randn_patched(size, *args, generator=None, **kwargs):
    """
    Acepta una lista de torch.Generator sin romper la API original.
    • Si generator es lista -> genera un tensor por seed y concatena en dim‑0
    • Si generator es None o torch.Generator -> llama a la versión original
    """
    # Caso normal → sin impacto en rendimiento
    if generator is None or isinstance(generator, torch.Generator):
        return _orig_randn(size, *args, generator=generator, **kwargs)

    # Caso especial: lista de generadores
    if isinstance(generator, (list, tuple)):
        tensors = []
        for g in generator:
            t = _orig_randn(size, *args, generator=g, **kwargs)
            tensors.append(t)
        return torch.stack(tensors, dim=0)  # crea el batch

    # Cualquier otro tipo (dict, int, etc.) ⇒ reproducir el error original
    return _orig_randn(size, *args, generator=generator, **kwargs)

# Activar el parche
torch.randn = _randn_patched
print("✅ Parche randn() multi‑generator activado")
