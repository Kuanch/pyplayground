import argparse

import picamera


def record(minutes):
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        camera.framerate = 20
        camera.vflip = True
        camera.start_recording('test.h264', quality=20, bitrate=1000000)
        camera.wait_recording(60 * minutes)
        camera.stop_recording()


def main(args):
    record(args.minutes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes')
    args = parser.parse_args()
    main(args)
