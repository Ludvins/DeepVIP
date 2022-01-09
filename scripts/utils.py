#!/usr/bin/env python


def create_file_name(args):
    filename = ""

    filename += "dataset=" + args.dataset_name  + "_"
    filename += "vip_layers=" + "-".join(str(i) for i in args.vip_layers)
    filename += "epochs=" + str(args.epochs) + "_"
    filename += "dropout=" + str(args.dropout) + "_"
    filename += "lr=" + str(args.lr) + "_"
    filename += "genf=" + "BNN_bnn-structure=" + "-".join(str(i) for i in args.bnn_structure)
        if args.genf == "BNN"
        else "BNN-GP_inner-dim=" + str(args.bnn_inner_dim)
    filename += "alpha=" + str(args.bb_alpha)
    filename += "prior_kl=" + "True" if args.prior_kl else "False" + "_"
    filename += "zero_mean_prior=" + "True" if args.zero_mean_prior else "False" + "_"
    filename += "prior_fixed_noise=" + "True" if args.fix_prior_noise else "False" + "_"
    filename += "split=" + str(args.split)

    return filename
