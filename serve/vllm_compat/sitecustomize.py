"""Process-wide compatibility patches for the RASST vLLM runtime."""

try:
    from transformers import AutoConfig

    _original_register = AutoConfig.register

    def _safe_register(model_type, config, exist_ok=False):
        try:
            return _original_register(model_type, config, exist_ok=exist_ok)
        except ValueError as exc:
            if model_type == "aimv2" and "already used" in str(exc):
                return None
            raise

    AutoConfig.register = _safe_register
except Exception:
    pass
