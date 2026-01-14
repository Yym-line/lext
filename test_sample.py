import wave

with wave.open("/Share/yym/node/S0_16/SEF-PNet-main/test/enroll/1-1.wav", "rb") as w:
    print("采样率:", w.getframerate())
