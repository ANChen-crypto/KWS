import serial
from serial.tools import list_ports
import numpy as np
import sounddevice as sd
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa
from time import time_ns
from modeling import CNN
import torch

model = CNN(input_dim=35)
model.load_state_dict(torch.load("./weights/cnn.pth", map_location="cpu")["model"], strict=True)
model.eval()

model(torch.randn(1, 35, 197))

def calc_checksum(audio):
    checksum = 0
    data = audio.astype(np.uint16)

    for x in data:
        checksum += int(x)

    return checksum & 0xFFFFFFFF

start_tag = b"[###LiveMore-Audiowav start###]\r\n"
end_tag   = b"[###LiveMore-Audiowav end###]\r\n"

ports = list(list_ports.comports())
for p in ports:
    if (p.serial_number == "CDC_ACM001"):
        pn = p.name
ser = serial.Serial(pn, 115200, timeout=1)

state = "WAIT_START"
buffer = bytearray()
audio_buffer = bytearray()
saveaudio = np.array([], dtype=np.int16)

ser.write(b"usb.start_audio_transmit\r\n")
file_idx = 0
date = "0911"
t_prev = 0
try:
    while True:

        buffer += ser.read(512)

        if state == "WAIT_START":

            idx = buffer.find(start_tag)

            if idx == -1:
                buffer = buffer[-len(start_tag):]
                continue

            buffer = buffer[idx + len(start_tag):]

            state = "READ_CHECKSUM"
            continue


        if state == "READ_CHECKSUM":

            idx = buffer.find(b"\r\n")

            if idx == -1:
                continue

            line = buffer[:idx].decode()

            if line.startswith("checksum="):
                expected_checksum = int(
                    line.split("=")[1],
                    16
                )

                # print(
                #     "expect checksum:",
                #     hex(expected_checksum)
                # )

            buffer = buffer[idx+2:]

            audio_buffer = bytearray()

            state = "RECV_AUDIO"
            continue


        if state == "RECV_AUDIO":

            while len(audio_buffer) < 6400:

                if len(buffer) == 0:
                    buffer += ser.read(512)
                    continue

                need = 6400 - len(audio_buffer)

                if len(buffer) <= need:
                    audio_buffer += buffer
                    buffer = bytearray()
                else:
                    audio_buffer += buffer[:need]
                    buffer = buffer[need:]


            # print("audio bytes =", len(audio_buffer))


            audio = np.frombuffer(
                audio_buffer,
                dtype=np.int16
            )


            checksum = calc_checksum(audio)

            t_prev = time_ns()

            saveaudio = np.concatenate((saveaudio, audio))

            if len(saveaudio) >= 16000 * 2:
                mel_spec = librosa.feature.melspectrogram(
                    y=saveaudio / 32768,
                    sr=16000,
                    n_fft=512,
                    hop_length=160,
                    n_mels=40,
                    center=False,
                )
                mel_spec = librosa.power_to_db(mel_spec)[None, :35, :]
                with torch.no_grad():
                    out = model(torch.tensor(mel_spec, dtype=torch.float32))
                    probs = F.softmax(out["logits"], dim=-1).squeeze().numpy()
                print(np.argmax(probs), max(probs).item())
                saveaudio = saveaudio[3200:]

            ser.write(b"usb.host_ack\r\n")

            audio_buffer = bytearray()
            state = "WAIT_START"
            continue
except KeyboardInterrupt:
    ser.write(b"usb.stop_audio_transmit\r\n")