from .trans_vg_ca import TransVG_ca, TransVG


def build_model(args):
    if args.model_name == 'TransVG_ca':
        return TransVG_ca(args)
    elif args.model_name == 'TransVG':
        return TransVG(args)
