"""Compatibility patches for third-party library version mismatches."""

import transformers.utils.import_utils as _tiu

# FlagEmbedding 1.3.5 imports is_torch_fx_available which was removed in
# transformers >= 4.58. Patch it back as a no-op so the import doesn't crash.
if not hasattr(_tiu, "is_torch_fx_available"):
    _tiu.is_torch_fx_available = lambda: False
