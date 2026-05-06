"""
Base class for all behaviour modules.
"""

class BaseModule:
    def __init__(self, dog, speech, priority):
        self.dog      = dog
        self.speech   = speech
        self.priority = priority
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def should_act(self, sensor_data: dict) -> bool:
        """Return True if this module wants to take control this tick."""
        raise NotImplementedError

    def act(self, sensor_data: dict):
        """Execute one tick of this module's behaviour."""
        raise NotImplementedError
