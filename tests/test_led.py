#!/usr/bin/env python3
"""
test_led.py — standalone GPIO LED test / wiring check for the Raspberry Pi.

Two modes, selected with --mode:

    single   drive a single LED on GPIO 17 (default). Tries gpiozero, then a
             gpiozero mock factory (for testing off-hardware), then raw
             RPi.GPIO — and prints clear guidance if none of them work.
    binary   display a value 0-15 as a 4-bit pattern on the LED array
             (GPIO 17/27/22/23 = bits 0/1/2/3). Enter 0-15 to show a value,
             16 to cycle through all of them, 'q' to quit.

Usage (run from the project root):
    python tests/test_led.py                 # single-LED test on GPIO 17
    python tests/test_led.py --mode binary   # 4-LED binary display

GPIO access usually needs sudo:  sudo python3 tests/test_led.py

Wiring (active-LOW):  3.3 V → LED anode, LED cathode → ~220 Ω resistor → GPIO
                      pin. The pin sinks current, so a bit is pulled LOW to turn
                      it on. Single mode uses GPIO 17; binary mode uses
                      GPIO 17/27/22/23.

NOTE: the binary display drives a pin LOW for a `1` bit and HIGH for a `0` — the
same active-LOW convention as run.py's BinaryLEDController, so this wiring check
matches the live feed. See the README ("GPIO LED output").
"""

import argparse
import select
import sys
import time
import warnings

SINGLE_PIN = 17
BINARY_PINS = [17, 27, 22, 23]   # 17 = LSB (2^0) … 23 = MSB (2^3)


def run_single() -> None:
    """Blink a single LED on GPIO 17 using whichever GPIO backend works."""
    print(f"\n  Testing single LED on GPIO {SINGLE_PIN}...")
    print("  Make sure the LED is connected to GPIO 17 (via a resistor) and GND.\n")

    # 1. gpiozero with the real pin factory.
    try:
        from gpiozero import LED

        led = LED(SINGLE_PIN, active_high=False)
        print(f"  ✓ LED object created on pin {SINGLE_PIN}")
        led.on()
        print("  ✓ LED ON — check the LED is lit (holding for 5 s)...")
        time.sleep(5)
        led.off()
        led.close()
        print("  ✓ LED OFF — GPIO cleaned up")
        print("  If the LED lit up, GPIO is working correctly!\n")
        return
    except PermissionError:
        print(f"  ✗ Permission denied — try:  sudo python3 {sys.argv[0]}\n")
        return
    except Exception as exc:
        print(f"  [1] gpiozero failed: {exc}")
        print("  [2] Trying gpiozero mock factory (simulated)...")

    # 2. gpiozero mock factory — verifies the logic without hardware.
    try:
        warnings.filterwarnings("ignore")
        from gpiozero import Device, LED
        from gpiozero.pins.mock import MockFactory

        Device.pin_factory = MockFactory()
        led = LED(SINGLE_PIN, active_high=False)
        led.on()
        print("  ✓ Mock GPIO ok (simulated — no real hardware touched)")
        time.sleep(1)
        led.off()
        Device.pin_factory.reset()
        return
    except Exception as exc:
        print(f"  [2] mock factory failed: {exc}")
        print("  [3] Trying RPi.GPIO directly...")

    # 3. Raw RPi.GPIO.
    try:
        import RPi.GPIO as GPIO

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SINGLE_PIN, GPIO.OUT)
        print(f"  ✓ RPi.GPIO initialised on pin {SINGLE_PIN}")
        GPIO.output(SINGLE_PIN, GPIO.LOW)   # active-LOW: LOW = on
        print("  ✓ LED ON for 5 s...")
        time.sleep(5)
        GPIO.output(SINGLE_PIN, GPIO.HIGH)
        print("  ✓ LED OFF — GPIO cleaned up\n")
        return
    except PermissionError:
        print(f"  ✗ Permission denied — try:  sudo python3 {sys.argv[0]}\n")
    except Exception as exc:
        print(f"  ✗ RPi.GPIO failed: {exc}\n")

    print("  ✗ Could not initialise GPIO. Checklist:")
    print(f"    1. Run with sudo:  sudo python3 {sys.argv[0]}")
    print(f"    2. LED connected to GPIO {SINGLE_PIN} + 220Ω resistor + GND")
    print("    3. RPi.GPIO / gpiozero installed for this Python\n")


def run_binary() -> None:
    """Display 0-15 as a 4-bit pattern; 16 cycles, 'q' quits."""
    try:
        import RPi.GPIO as GPIO
    except Exception as exc:
        print(f"  ✗ RPi.GPIO unavailable: {exc}")
        print(f"    Run with:  sudo python3 {sys.argv[0]} --mode binary\n")
        return

    GPIO.setmode(GPIO.BCM)
    for pin in BINARY_PINS:
        GPIO.setup(pin, GPIO.OUT)

    def display(number: int) -> None:
        binary = format(number, "04b")  # 4-bit string
        print(f"  Displaying {number} in binary: {binary}")
        for i, bit in enumerate(reversed(binary)):  # reversed to match pin order
            GPIO.output(BINARY_PINS[i], GPIO.LOW if bit == "1" else GPIO.HIGH)

    def off() -> None:
        for pin in BINARY_PINS:
            GPIO.output(pin, GPIO.HIGH)

    print("  Binary LED display — GPIO 17=bit0, 27=bit1, 22=bit2, 23=bit3")
    print(f"  Pins: {BINARY_PINS}")
    print("  Enter 0-15 to display, 16 to cycle through all values, 'q' to quit.\n")

    number = 0
    cycling = False
    try:
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline().strip()
                if line.lower() == "q":
                    break
                try:
                    value = int(line)
                except ValueError:
                    print("  Invalid input — enter 0-15, 16, or 'q'.")
                    continue
                if 0 <= value <= 15:
                    cycling = False
                    number = value
                    display(number)
                elif value == 16:
                    cycling = True
                    number = 0
                else:
                    print("  Number must be between 0 and 15, or 16 to cycle.")
            elif cycling:
                display(number)
                time.sleep(1)
                number = (number + 1) % 16
            else:
                time.sleep(0.1)  # avoid hogging CPU
    except KeyboardInterrupt:
        print("\n  Interrupted...")
    finally:
        off()
        GPIO.cleanup()

    print("  Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone GPIO LED test / wiring check.")
    parser.add_argument(
        "--mode", choices=("single", "binary"), default="single",
        help="single-LED blink on GPIO 17, or 4-LED binary 0-15 display (default: single)",
    )
    args = parser.parse_args()

    if args.mode == "single":
        run_single()
    else:
        run_binary()


if __name__ == "__main__":
    main()
