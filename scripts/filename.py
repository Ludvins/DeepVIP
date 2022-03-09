def create_file_name(args):
    filename = ""

    filename += "dataset=" + args.dataset_name
    filename += "_vip-layers=" + "-".join(str(i) for i in args.vip_layers)
    filename += "_epochs=" + str(args.epochs)
    filename += "_batch=" + str(args.batch_size)
    if args.lr != 0.001:
        filename += "_lr=" + str(args.lr)

    if args.genf == "BNN":
        dims = "-".join(str(i) for i in args.bnn_structure)
        filename += "_genf=BNN_bnn-structure=" + dims
        if args.dropout != 0:
            filename += "_dropout=" + str(args.dropout)
        filename += "_act=" + str(args.activation_str)    
        
    elif args.genf == "GP":
        filename += "_genf=GP_inner-dim=" + str(args.bnn_inner_dim)
    else:
        filename += "_genf=" + args.genf

    filename += "_regression-coeffs=" + str(args.regression_coeffs)
    if args.bb_alpha != 0:
        filename += "_alpha=" + str(args.bb_alpha)

    if args.num_samples_train != 1:
        filename += "_num_samples_train=" + str(args.num_samples_train)

    if args.prior_kl:
        filename += "_prior-kl"
    if args.zero_mean_prior:
        filename += "_zero-mean-prior"
    if args.fix_prior_noise:
        filename += "_prior-fixed-noise"
    if args.freeze_ll:
        filename += "_freeze-ll-noise"

    if args.split is not None:
        filename += "_split=" + str(args.split)

    return filename
