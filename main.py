import argparse
import os

from src.CPP.Environment import CPPEnvironmentParams, CPPEnvironment
from src.DH.Environment import DHEnvironmentParams, DHEnvironment
from src.DHMulti.Environment import DHMultiEnvironment

from utils import *


def main_cpp(p):
    env = CPPEnvironment(p)

    env.run()


def main_dh(p):
    env = DHEnvironment(p)

    env.run()


def main_dh_multi(p):
    env = DHMultiEnvironment(p)

    env.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=True, action='store_true', help='Activates usage of GPU')
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    parser.add_argument('--config', default='config/充电桩位置固定map变化.json', help='Path to config file')
    parser.add_argument('--id', default='测试2e6', help='If set overrides the logfile name and the save name')
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--cpp', action='store_true', help='Run Coverage Path Planning')
    parser.add_argument('--dh', default=True, action='store_true', help='Run Path Planning for Data Harvesting')
    parser.add_argument('--multi', action='store_true', help='Run Path Planning for Multi (So far only DH)')

    parser.add_argument('--battery')
    args = parser.parse_args()

    if args.generate_config:
        if args.cpp:
            generate_config(CPPEnvironmentParams(), "config/cpp.json")
        elif args.dh:
            generate_config(DHEnvironmentParams(), "config/dh.json")
        else:
            print("Specify which config to generate, DH or CPP")
        exit(0)

    if args.config is None:
        print("Config file needed!")
        exit(1)

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params = read_config(args.config)

    if args.params is not None:
        params = override_params(params, args.params)

    if args.id is not None:
        params.model_stats_params.save_model = "models/" + args.id
        params.model_stats_params.log_file_name = args.id

    if args.cpp:
        main_cpp(params)
    elif args.dh:
        if args.multi:
            main_dh_multi(params)
        else:
            main_dh(params)
