from enum import Enum, auto
import time

class safety(Enum):
    NORMAL = auto()
    WARNING = auto()
    ALARM = auto()

class safety_controller:
    def __init__(self, config=None):
        self.config = config or {
            'normal_threshold': 0.5,
            'warning_threshold': 0.8,
            'alarm_threshold': 0.9
        }
        self.state = safety.NORMAL
        self.last_state_change = time.time()
        self.state_change_count = 0

    def update(self, score):
        if score < self.config['normal_threshold']:
            self.state = safety.NORMAL
        elif score < self.config['warning_threshold']:
            self.state = safety.WARNING
        elif score < self.config['alarm_threshold']:
            self.state = safety.ALARM
        else:
            self.state = safety.ALARM
        self.state_change_count += 1
        self.last_state_change = time.time()

    def get_state(self):
        return self.state

    def get_state_change_count(self):
        return self.state_change_count

    def get_last_state_change(self):
        return self.last_state_change

    def get_action(self):
        if self.state == safety.NORMAL:
            return 'normal'
        elif self.state == safety.WARNING:
            return 'warning'
        elif self.state == safety.ALARM:
            return 'alarm'