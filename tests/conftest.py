import sys
import types
from unittest.mock import MagicMock

# Setup mock modules once for the entire test session
if 'mlx' not in sys.modules:
    mlx = types.ModuleType('mlx')
    mlx_core = types.ModuleType('mlx.core')
    mlx_core_metal = types.ModuleType('mlx.core.metal')
    mlx_whisper = types.ModuleType('mlx_whisper')
    mlx_whisper_transcribe = types.ModuleType('mlx_whisper.transcribe')
    mlx_whisper_load_models = types.ModuleType('mlx_whisper.load_models')

    sys.modules['mlx'] = mlx
    sys.modules['mlx.core'] = mlx_core
    mlx.core = mlx_core
    sys.modules['mlx.core.metal'] = mlx_core_metal
    mlx_core.metal = mlx_core_metal

    sys.modules['mlx_whisper'] = mlx_whisper
    sys.modules['mlx_whisper.transcribe'] = mlx_whisper_transcribe
    mlx_whisper.transcribe = mlx_whisper_transcribe
    sys.modules['mlx_whisper.load_models'] = mlx_whisper_load_models
    mlx_whisper.load_models = mlx_whisper_load_models

    # Add default mocks / classes
    mlx_core_metal.clear_cache = MagicMock(name='clear_cache')
    mlx_whisper_transcribe.transcribe = MagicMock(name='transcribe')
    mlx_whisper_load_models.load_model = MagicMock(name='load_model')

    class MockModelHolder:
        model = "some_model"
        model_path = "some_path"

    mlx_whisper_transcribe.ModelHolder = MockModelHolder
