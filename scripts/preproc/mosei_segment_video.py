import ffmpeg
import os
import argparse
from tqdm import tqdm


def _parse():
    parser = argparse.ArgumentParser(description='get utterance-level video/audio')
    parser.add_argument("fullvideo_dir", type=str, help='CMU-MOSEI full video path')
    parser.add_argument("interval_list", type=str, help='CMU-MOSEI interval list')
    parser.add_argument("segvideo_dir", type=str, help='CMU-MOSEI segmented video path')
    parser.add_argument("segaudio_dir", type=str, help='CMU-MOSEI segmented audio path')
    args = parser.parse_args()
    return args


def _main():
    args = _parse()
    with open(args.interval_list, "r") as fp:
        lines = fp.readlines()

    os.makedirs(args.segvideo_dir, exist_ok=True)

    for line in tqdm(lines):
        filename, clipnum, start, end = line.strip().split()
        if float(start) < 0:
            start = "0.0"

        in_video = os.path.join(args.fullvideo_dir, filename+'.mp4')
        out_video = os.path.join(args.segvideo_dir, filename+'_'+clipnum+'.mp4')
        out_audio = os.path.join(args.segaudio_dir, filename+'_'+clipnum+'.wav')
        if not os.path.exists(out_video) or not os.path.exists(out_audio):
            os.makedirs(os.path.dirname(out_video), exist_ok=True)
            os.makedirs(os.path.dirname(out_audio), exist_ok=True)
            stream = ffmpeg.input(in_video)
            v_stream = ffmpeg.output(
                    stream,
                    out_video,
                    ss=start,
                    to=end,
                    )
            a_stream = ffmpeg.output(
                    stream,
                    out_audio,
                    ar=16000,
                    ac=1,
                    f="wav",
                    ss=start,
                    to=end,
                    )
            ffmpeg.run(v_stream, quiet=True)
            ffmpeg.run(a_stream, quiet=True)


if __name__ == "__main__":
    _main()
