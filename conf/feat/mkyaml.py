import os
import yaml
yaml.Dumper.ignore_aliases = lambda *args : True


if __name__ == '__main__':
    outd = './'

    slice_dic = {'layer': 0}
    window_dic = {'proc': 'flatten', 'window_size': 1, 'window_shift': 1}
    specaug_dic = {
            'max_time_warp': 0,
            'max_freq_rate': 0.02,
            'n_freq_mask': 10,
            'max_time_rate': 0.2,
            'n_time_mask': 2,
            'total_max_time_rate': 0.4,
            }

    for layer in range(25):
        outf = os.path.join(outd, 'layer{}.yaml'.format(layer))
        slice_dic['layer'] = layer
        d = {
                'video_feat': {
                    'slice': slice_dic,
                    'sliding_window': window_dic,
                    'spec_augment': specaug_dic,
                    },
                'audio_feat': {
                    'slice': slice_dic,
                    'sliding_window': window_dic,
                    'spec_augment': specaug_dic,
                    },
                'text_feat': {
                    'slice': slice_dic,
                    'sliding_window': window_dic,
                    'spec_augment': specaug_dic,
                    },
                }

        with open(outf, 'w') as fp:
            yaml.dump(d, fp, default_flow_style=False)


