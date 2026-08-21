from hardware.plc import PLC
class SimulatedPLC(PLC):
    def __init__(self,cfg): self.connected=False; self.last_result=None
    def connect(self): self.connected=True
    def disconnect(self): self.connected=False
    def send_result(self,result):
        if not self.connected: raise RuntimeError("PLC not connected")
        self.last_result=result
