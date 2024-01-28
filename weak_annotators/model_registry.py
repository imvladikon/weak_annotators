__MODELS_REGISTRY__ = {}


class _ModelsRegistry:

    def get_model(self, model_name):
        return (
            __MODELS_REGISTRY__[model_name]
            if model_name in __MODELS_REGISTRY__
            else None
        )

    def register_model(self, model_name, model):
        __MODELS_REGISTRY__[model_name] = model
        return model

    def has(self, model_name):
        return model_name in __MODELS_REGISTRY__


ModelsRegistry = _ModelsRegistry()

__all__ = [
    "ModelsRegistry",
]
