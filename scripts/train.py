import argparse
import contextlib
import datetime
import os
import sys
from typing import Dict, Union

import yaml

from af3_binding.trainer import BDtrainer


def parse_config(config: str) -> Dict:
    assert isinstance(config, str)
    assert os.path.exists(config)
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    if not config['name']:
        config['name'] = datetime.datetime.now().strftime('%Y-%m-%d')
    return config

def main(config: Dict, **kwargs) -> None:
    assert isinstance(config, Dict)
    m_trainer = BDtrainer(config, **kwargs)
    m_trainer.train()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-c', '--config', type=str, default='./config/config.yml', help='path to config file')
    args = argparser.parse_args()
    config = parse_config(args.config)

    redirect_file = os.readlink(f'/proc/{os.getpid()}/fd/1')

    if sys.stdout.isatty() or os.path.basename(redirect_file) != 'nohup.out':
        # stdout is not redirected or not auto redirected by nohup 
        main(config)
    else:
        # redirect stdout to `output_dir/name.log`
        output_dir = config['output_dir']
        name = config['name']
        with open(os.path.join(output_dir, f'{name}.log'), 'w') as f:
            with contextlib.redirect_stdout(f):
                main(config)
            os.remove(redirect_file)
