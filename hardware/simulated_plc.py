from hardware.plc import PLC

class SimulatedPLC(PLC):
    def __init__(self, config):
        self.config = config
        self.connected = False
        self.last_result = None

    def connect(self):
        self.connected = True

    def send_result(self, result):
        if not self.connected:
            raise RuntimeError("Simulated PLC is not connected")
        self.last_result = result

    def disconnect(self):
        self.connected = False

    def is_connected(self):
        return self.connected
