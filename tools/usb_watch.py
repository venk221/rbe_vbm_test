#!/usr/bin/env python3

import argparse
import glob
import os
import subprocess
import sys
import threading
import time
from datetime import datetime


HELP_TEXT = ('Watches the D405 USB connection (no sudo needed) and flags autosuspend as a likely\n'
            'cause of flaky drops. See the comment above HELP_TEXT in this file for the admin-side fix.')


def find_device(vendor, product):
    for vendor_file in glob.glob('/sys/bus/usb/devices/*/idVendor'):
        device_dir = os.path.dirname(vendor_file)
        try:
            found_vendor = open(vendor_file).read().strip()
            found_product = open(os.path.join(device_dir, 'idProduct')).read().strip()
        except OSError:
            continue
        if found_vendor == vendor and found_product == product:
            return device_dir
    return None


def read(path, default='?'):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def ts():
    return datetime.now().strftime('%H:%M:%S.%f')[:-3]


def poll_power_state(path, stop_flag, log):
    last = None
    while not stop_flag['stop']:
        status = read(os.path.join(path, 'power', 'runtime_status')) if os.path.isdir(path) else 'GONE'
        if status != last:
            log(f'[{ts()}] power/runtime_status = {status}')
            last = status
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=HELP_TEXT, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vendor', default='8086', help='USB idVendor, hex without 0x (default: 8086, Intel)')
    parser.add_argument('--product', default='0b5b', help='USB idProduct, hex without 0x (default: 0b5b, D405)')
    parser.add_argument('--log', help='also append timestamped lines to this file')
    args = parser.parse_args()

    logfile = open(args.log, 'a') if args.log else None

    def log(line):
        print(line, flush=True)
        if logfile:
            logfile.write(line + '\n')
            logfile.flush()

    path = find_device(args.vendor, args.product)
    if path is None:
        log(f'No USB device {args.vendor}:{args.product} found right now -- plug in the camera first.')
        return 1
    control = read(os.path.join(path, 'power/control'))
    log(f'Watching {path} ({read(os.path.join(path, "product"))}), power/control={control}')
    if control == 'auto':
        log('  -> autosuspend is ENABLED for this device. That is a plausible contributor to '
            'intermittent drops; ask the admin account to set it to "on" (see this script\'s --help).')

    stop_flag = {'stop': False}
    poller = threading.Thread(target=poll_power_state, args=(path, stop_flag, log), daemon=True)
    poller.start()

    proc = subprocess.Popen(['udevadm', 'monitor', '--udev', '--subsystem-match=usb'],
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    log('udevadm monitor running -- leave this up during a capture session, Ctrl-C to stop.')
    try:
        for line in proc.stdout:
            line = line.strip()
            if line:
                log(f'[{ts()}] {line}')
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag['stop'] = True
        proc.terminate()
        if logfile:
            logfile.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
