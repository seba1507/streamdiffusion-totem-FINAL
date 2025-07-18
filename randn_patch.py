# randn_patch.py  v2
import torch
from functools import wraps

_orig_randn = torch.randn  # copia de la función original

@wraps(_orig_randn)
def _randn_patched(size, *args, generator=None, **kwargs):
    """
    Parche que permite:
    • generator = torch.Generator  -> se pasa tal cual
    • generator = list/tuple       -> genera un tensor por seed y concatena
    • generator = None | inválido  -> NO se pasa el argumento (evita el bug)
    """
    # Caso 1 ─ lista de generadores
    if isinstance(generator, (list, tuple)):
        tensors = [
            _orig_randn(size, *args, generator=g, **kwargs)
            for g in generator
        ]
        return torch.stack(tensors, dim=0)

    # Caso 2 ─ generador válido
    if isinstance(generator, torch.Generator):
        return _orig_randn(size, *args, generator=generator, **kwargs)

    # Caso 3 ─ None u otro valor no soportado → llamamos SIN el kwarg
    return _orig_randn(size, *args, **kwargs)

# Activar parche
torch.randn = _randn_patched
print("✅ Parche randn() multi‑generator v2 activado")
