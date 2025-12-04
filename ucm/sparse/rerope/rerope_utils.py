import os

class VllmPatchConfig:
    
    ENV_VAR_NAME = "VLLM_USE_REROPE"
    DEFAULT_REROPE_WINDOW = 32768
    DEFAULT_TRAINING_LENGTH = 32768
    
    def __init__(self):
        self.use_rerope = False

        # the model's pre-training length 
        self.rerope_window = self.DEFAULT_REROPE_WINDOW
        self.training_length = self.DEFAULT_TRAINING_LENGTH
    
    @property
    def is_rerope_enabled(self) -> bool:
        env_val = os.getenv(self.ENV_VAR_NAME)
        if env_val is not None:
            return env_val.lower() in ("true", "1", "yes", "on")
        return self.use_rerope
    
    def update_rerope_params(self, rerope_window: int = None, training_length: int = None):
        """
        window: rerope's window size -> w
        training_len: the model's pre-training length
        window = training_len
        """
        if rerope_window is not None:
            self.rerope_window = rerope_window
        if training_length is not None:
            self.training_length = training_length
    
    def get_config_summary(self) -> dict:
        return {
            "use_rerope": self.is_rerope_enabled,
            "rerope_window": self.rerope_window,
            "training_length": self.training_length,
            "env_var": f"{self.ENV_VAR_NAME}={os.getenv(self.ENV_VAR_NAME, 'not set')}",
        }


_default_config = VllmPatchConfig()