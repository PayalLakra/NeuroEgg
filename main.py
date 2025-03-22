import asyncio
from bleak import BleakScanner, BleakClient
from pylsl import StreamInfo, StreamOutlet
import numpy as np
from scipy.signal import butter, lfilter
import time
from pynput.keyboard import Key, Controller

# Constants
DEVICE_NAME = "NPG-30:30:f9:f9:db:76"
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CONTROL_CHAR_UUID = "0000ff01-0000-1000-8000-00805f9b34fb"

keyboard = Controller()

SINGLE_SAMPLE_LEN = 7  # Each sample is 7 bytes
BLOCK_COUNT = 10  # Batch size: 10 samples per notification
NEW_PACKET_LEN = SINGLE_SAMPLE_LEN * BLOCK_COUNT
stream_name = "NPG"
info = StreamInfo(stream_name, "EXG", 3, 500, "int16", "uid007")
outlet = StreamOutlet(info)
sampling_rate = 500  # Sampling rate in Hz
buffer_size = int(sampling_rate * 0.2)  # 0.2 seconds buffer (smaller for real-time)

# Filter parameters
b, a = butter(4, 70.0 / (0.5 * sampling_rate), btype='high')
rms_window_size = int(0.1 * sampling_rate)  # 0.1 seconds window

# Class for processing and printing envelope for a specific channel
class ChannelProcessor:
    def __init__(self, channel_index, buffer_size, filter_coeffs, window_size):
        self.channel_index = channel_index
        self.buffer = np.zeros(buffer_size)
        self.current_index = 0
        self.b, self.a = filter_coeffs
        self.window_size = window_size
        self.last_print_time = time.time()
        self.print_interval = 0.05  # Print every 0.05 seconds (20 Hz)
        self.last_trigger_time = 0  # Track the last time a command was triggered
        self.debounce_interval = 0.5  # Minimum time between commands (in seconds)

    def calculate_moving_rms(self, signal, window_size):
        # Efficient RMS calculation using a rolling window
        squared_signal = signal**2
        window = np.ones(window_size) / window_size
        rms = np.sqrt(np.convolve(squared_signal, window, mode='valid'))
        return np.pad(rms, (len(signal) - len(rms), 0), 'constant')

    def process_sample(self, sample_value):
        # Update circular buffer
        self.buffer[self.current_index] = sample_value
        self.current_index = (self.current_index + 1) % len(self.buffer)

        # Apply filter
        filtered_signal = lfilter(self.b, self.a, self.buffer)

        # Calculate RMS envelope
        abs_filtered_signal = np.abs(filtered_signal)
        rms_envelope = self.calculate_moving_rms(abs_filtered_signal, self.window_size)

        # Print envelope value at the specified interval
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_interval:
            # print(f"Channel {self.channel_index} Envelope: {rms_envelope[-1]:.2f}")
            if rms_envelope[-1] > 200:
                # Check if enough time has passed since the last command
                if current_time - self.last_trigger_time >= self.debounce_interval:
                    if self.channel_index == 2:
                        keyboard.press (Key.right )
                        time.sleep(0.02)
                        keyboard.release (Key.right )
                    elif self.channel_index == 3:
                        keyboard.press (Key.left )
                        time.sleep(0.02)
                        keyboard.release (Key.left )
                    self.last_trigger_time = current_time  # Update the last trigger time
            self.last_print_time = current_time

# Create objects for Channel 2 and Channel 3
channel2_processor = ChannelProcessor(channel_index=2, buffer_size=buffer_size, filter_coeffs=(b, a), window_size=rms_window_size)
channel3_processor = ChannelProcessor(channel_index=3, buffer_size=buffer_size, filter_coeffs=(b, a), window_size=rms_window_size)

def process_sample(sample_data: bytearray):
    # Extract channel values from the sample
    channels = [
        int.from_bytes(sample_data[1:3], byteorder='big', signed=True),  # Channel 1
        int.from_bytes(sample_data[3:5], byteorder='big', signed=True),  # Channel 2
        int.from_bytes(sample_data[5:7], byteorder='big', signed=True)   # Channel 3
    ]

    channel2_processor.process_sample(channels[1])
    channel3_processor.process_sample(channels[2])

def notification_handler(sender, data: bytearray):
    if len(data) == NEW_PACKET_LEN:
        for i in range(0, NEW_PACKET_LEN, SINGLE_SAMPLE_LEN):
            sample = data[i:i + SINGLE_SAMPLE_LEN]
            process_sample(sample)
    elif len(data) == SINGLE_SAMPLE_LEN:
        process_sample(data)
    else:
        print("Unexpected packet length:", len(data))

async def run():
    print("Scanning for BLE devices with name starting with", DEVICE_NAME)
    devices = await BleakScanner.discover()

    # Find the target device
    target = None
    for d in devices:
        if d.name and DEVICE_NAME.lower() in d.name.lower():
            target = d
            break
    if target is None:
        print("No target device found")
        return

    print("Connecting to:", target.name, target.address)
    async with BleakClient(target) as client:
        if not client.is_connected:
            print("Failed to connect")
            return

        print("Connected to", target.name)
        await client.write_gatt_char(CONTROL_CHAR_UUID, b"START", response=True)
        print("Sent START command")
        await client.start_notify(DATA_CHAR_UUID, notification_handler)
        print("Subscribed to data notifications")
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(run())