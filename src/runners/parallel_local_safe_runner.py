import time

from .parallel_runner import ParallelRunner


class ParallelLocalSafeRunner(ParallelRunner):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self._did_initial_safe_reset = False

    def reset(self):
        if self._did_initial_safe_reset:
            return super().reset()

        self.batch = self.new_batch()
        pre_transition_data = self._new_pre_transition_data()

        stagger_sec = getattr(self.args, "env_reset_stagger_sec", 0.25)

        # Serialize reset/launch on local macOS to avoid multiple SC2 instances
        # racing on the same TempLaunchMap.SC2Map.
        for idx, parent_conn in enumerate(self.parent_conns):
            parent_conn.send(("reset", None))
            data = parent_conn.recv()
            self._append_reset_pre_transition(pre_transition_data, data)
            if stagger_sec > 0 and idx != len(self.parent_conns) - 1:
                time.sleep(stagger_sec)

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0
        self._did_initial_safe_reset = True
