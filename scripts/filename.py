
def create_file_name(args):
    filename = ""

    filename += "dataset=" + args.dataset_name  + "_"
    filename += "vip_layers=" + "-".join(str(i) for i in args.vip_layers) + "_"
    filename += "epochs=" + str(args.epochs) + "_"
    filename += "dropout=" + str(args.dropout) + "_"
    filename += "lr=" + str(args.lr) + "_"
    filename += "genf=" + "BNN_bnn-structure=" + "-".join(str(i) for i in args.bnn_structure) if args.genf == "BNN" else "BNN-GP_inner-dim=" + str(args.bnn_inner_dim)+ "_"
    filename += "alpha=" + str(args.bb_alpha)+ "_"
    filename += "prior_kl_" if args.prior_kl else ""
    filename += "zero_mean_prior_" if args.zero_mean_prior else ""
    filename += "prior_fixed_noise_" if args.fix_prior_noise else ""
    filename += "split=" + str(args.split)

    return filename
