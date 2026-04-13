REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .parallel_local_safe_runner import ParallelLocalSafeRunner
REGISTRY["parallel_local_safe"] = ParallelLocalSafeRunner
