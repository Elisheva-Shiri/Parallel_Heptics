"""Quick utility to list all serial ports on this computer.

Run with:
    uv run list_ports.py

Useful for finding which COM port your ESP32 is on.
"""

from serial.tools import list_ports


# Common USB-to-serial chips used on ESP32 dev boards.
# (vid, pid) -> friendly name
KNOWN_ESP32_CHIPS: dict[tuple[int, int], str] = {
    (0x10C4, 0xEA60): "Silicon Labs CP210x (common on ESP32 DevKit)",
    (0x1A86, 0x7523): "QinHeng CH340 (common on ESP32 clones)",
    (0x1A86, 0x55D4): "QinHeng CH9102",
    (0x303A, 0x1001): "Espressif native USB (ESP32-S2/S3/C3)",
    (0x0403, 0x6010): "FTDI FT2232 (some ESP32 boards)",
    (0x0403, 0x6001): "FTDI FT232 (some ESP32 boards)",
}


def main() -> None:
    ports = list(list_ports.comports())

    if not ports:
        print("No serial ports detected.")
        print("Tips:")
        print("  - Plug the ESP32 in with a *data* USB cable (not charge-only).")
        print("  - Check Windows Device Manager for 'Unknown device'.")
        print("  - Install the CP210x or CH340 USB driver if needed.")
        return

    print(f"Found {len(ports)} serial port(s):\n")
    for p in ports:
        vidpid = f"{p.vid:04X}:{p.pid:04X}" if p.vid is not None else "----:----"
        guess = ""
        if p.vid is not None and (p.vid, p.pid) in KNOWN_ESP32_CHIPS:
            guess = f"  <-- likely ESP32 ({KNOWN_ESP32_CHIPS[(p.vid, p.pid)]})"

        print(f"  {p.device}")
        print(f"      description : {p.description}")
        print(f"      manufacturer: {p.manufacturer}")
        print(f"      VID:PID     : {vidpid}{guess}")
        print(f"      serial #    : {p.serial_number}")
        print(f"      hwid        : {p.hwid}")
        print()


if __name__ == "__main__":
    main()
