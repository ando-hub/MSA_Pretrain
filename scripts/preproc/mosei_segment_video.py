import ffmpeg
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool


def _parse():
    parser = argparse.ArgumentParser(description='get utterance-level video/audio')
    parser.add_argument("fullvideo_dir", type=str, help='CMU-MOSEI full video path')
    parser.add_argument("interval_list", type=str, help='CMU-MOSEI interval list')
    parser.add_argument("segvideo_dir", type=str, help='CMU-MOSEI segmented video path')
    parser.add_argument("segaudio_dir", type=str, help='CMU-MOSEI segmented audio path')
    parser.add_argument("-n", metavar='num_procs', type=int, default=1,
                        help='Num. of parallel process')
    args = parser.parse_args()
    return args


def segment_video(in_video, out_video, out_audio, start, end):
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


def _wrap_segment_video(args):
    return segment_video(*args)


def _main():
    args = _parse()

    # prepare argument list for parallel process
    arglist = []
    for line in open(args.interval_list):
        filename, clipnum, start, end = line.strip().split()
        if float(start) < 0:
            start = "0.0"
        in_video = os.path.join(args.fullvideo_dir, filename+'.mp4')
        out_video = os.path.join(args.segvideo_dir, filename+'_'+clipnum+'.mp4')
        out_audio = os.path.join(args.segaudio_dir, filename+'_'+clipnum+'.wav')
        if not os.path.exists(out_video) or not os.path.exists(out_audio):
            os.makedirs(os.path.dirname(out_video), exist_ok=True)
            os.makedirs(os.path.dirname(out_audio), exist_ok=True)
            arglist.append(
                    (in_video, out_video, out_audio, start, end)
                    )

    # run
    print('start video segmentation (num. parallel: {})'.format(args.n))
    pool = Pool(processes=args.n)
    with tqdm(total=len(arglist)) as t:
        for _ in pool.imap_unordered(_wrap_segment_video, arglist):
            t.update(1)


if __name__ == "__main__":
    _main()
