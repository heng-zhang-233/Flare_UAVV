import socket
import struct
import sys
import time

class MessagingClient:
    def __init__(self, host : str, port : int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))

    def request(self, data : bytes) -> bytes:
        self.socket.send(data)
        return self.socket.recv(1024)
    
    def close(self):
        self.socket.close()

class Controller:
    def __init__(self, host : str, port : int):
        self.messaging = MessagingClient(host, port)

    def get_control_index(self, name : str) -> int:
        request = b'\x00'
        array = name.encode("utf-8")
        request += int.to_bytes(len(array), length=1, byteorder=sys.byteorder, signed=False)
        request += array
        reply = self.messaging.request(request)
        if reply[0:1] != b'\x01':
            return None
        else:
            return int.from_bytes(reply[1:5], byteorder=sys.byteorder, signed=True)

    def get_time_step(self) -> float:
        reply = self.messaging.request(b'\x01')
        return struct.unpack('d', reply)[0]
    
    def simulate(self, steps : int, controls : dict) -> tuple:
        request = b'\x02'
        request += int.to_bytes(steps, length=1, byteorder=sys.byteorder, signed=False)
        request += int.to_bytes(len(controls), length=1, byteorder=sys.byteorder, signed=False)
        for hash in controls:
            request += int.to_bytes(hash, length=4, byteorder=sys.byteorder, signed=True)
            request += struct.pack('d', controls[hash])

        reply = self.messaging.request(request)
        if reply[0:1] != b'\x01':
            return None
        else:
            return struct.unpack('dddddddddddd', reply[1:])
    
    def reset(self):
        request = b'\x03'
        reply = self.messaging.request(request)

    def close(self):
        self.messaging.close()