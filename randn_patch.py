# randn_patch.py  v4 — autónomo (compatible con diffusers 0.24)
import torch
from functools import wraps

# --- copia de la randn original ---
_orig_randn = torch.randn  # NO se toca después

# --- mini‑implementación local ----
def _local_randn_tensor(shape, *, generator=None, device=None, dtype=None):
    """
    Imita diffusers.utils.randn_tensor para los casos que nos interesan.
    • shape: tuple[int]
    • generator: None | torch.Generator | list[torch.Generator]
    """
    # (A) una lista de generadores  → un tensor por seed
    if isinstance(generator, (list, tuple)):
        tensors = [
            _local_randn_tensor(shape, generator=g, device=device, dtype=dtype)
            for g in generator
        ]
        return torch.stack(tensors, dim=0)

    # (B) generador único o None
    if generator is not None and not isinstance(generator, torch.Generator):
        # tipo inválido → ignóralo y deja que _orig_randn lance su propio error
        generator = None

    return _orig_randn(
        shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )

# --------- parche global ----------
@wraps(_orig_randn)
def _randn_patched(shape, *args, generator=None, **kwargs):
    """
    Llama siempre a _local_randn_tensor, que entiende listas, None y generadores.
    Todos los casos anteriores del bug quedan cubiertos.
    """
    # extraemos device / dtype si se pasaron como kwargs o args
    device = kwargs.pop("device", None)
    dtype  = kwargs.pop("dtype",  None)

    # _local_randn_tensor maneja shape, generator, device, dtype
    return _local_randn_tensor(
        shape, generator=generator, device=device, dtype=dtype
    )

torch.randn = _randn_patched
print("✅ Parche randn() v4 autónomo activado")
