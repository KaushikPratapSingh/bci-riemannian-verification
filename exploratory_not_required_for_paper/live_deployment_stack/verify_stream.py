import asyncio
import struct
import time
from bleak import BleakScanner, BleakClient

# Match the exact UUIDs from your ESP32-S3 firmware code
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

class StreamVerifier:
    def __init__(self):
        self.last_packet_id = None
        self.packet_count = 0
        self.start_time = None
        self.dropped_packets = 0

    def notification_handler(self, characteristic, data):
        """
        Processes incoming live BLE packets.
        Unpacks the strict 14-byte memory block using Python's struct module.
        """
        if self.start_time is None:
            self.start_time = time.time()

        # Check payload size immediately
        if len(data) != 14:
            print(f"❌ Corrupted Frame! Expected 14 bytes, received {len(data)} bytes.")
            return

        # Format breakdown:
        # B = uint8_t (Packet ID)
        # h = int16_t (Channels 1-4)
        # I = uint32_t (Hardware timestamp)
        # B = uint8_t (Status footer)
        # '<' ensures Little-Endian unpacking (standard for ESP32)
        packet_id, ch1, ch2, ch3, ch4, hw_time, status = struct.unpack("<BhhhhIB", data)

        # Drop-out Tracking
        if self.last_packet_id is not None:
            # Handle standard uint8 wrap-around (255 -> 0)
            expected = (self.last_packet_id + 1) % 256
            if packet_id != expected:
                loss = (packet_id - expected) % 256
                self.dropped_packets += loss
                print(f"⚠️ Gap detected! Missed {loss} frame(s).")

        self.last_packet_id = packet_id
        self.packet_count += 1

        # Print diagnostics out to terminal every 250 samples (~once per second)
        if self.packet_count % 250 == 0:
            elapsed = time.time() - self.start_time
            true_fps = self.packet_count / elapsed
            print(f"📡 [LIVE] Sample Rate: {true_fps:.2f}Hz | Total Received: {self.packet_count} | Dropped: {self.dropped_packets} | Ch1 Raw: {ch1}")

async def main():
    print("Searching for 'BCI_Manifold_Headband'...")
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: d.name and "BCI_Manifold_Headband" in d.name
    )
    
    if not device:
        print("❌ Could not find headband. Make sure the ESP32-S3 is powered on and advertising.")
        return

    print(f"Found headband at {device.address}. Connecting...")
    verifier = StreamVerifier()

    async with BleakClient(device) as client:
        print("✅ Connected! Starting notification stream...")
        
        # Subscribe to the data stream characteristic
        await client.start_notify(CHARACTERISTIC_UUID, verifier.notification_handler)
        
        # Keep running and streaming for 30 seconds
        await asyncio.sleep(30.0)
        
        # Shutdown cleanly
        await client.stop_notify(CHARACTERISTIC_UUID)
        print("\n🏁 Test complete. Connection closed.")

if __name__ == "__main__":
    asyncio.run(main())