def create_file_name(args):
    filename = ""

    filename += "dataset=" + args.dataset_name

    filename += "_iterations=" + str(args.iterations)
    filename += "_batch=" + str(args.batch_size)
    if args.lr != 0.001:
        filename += "_lr=" + str(args.lr)

    filename += "_layer=" + args.bnn_layer_str
    filename += "_num_inducing=" + str(args.num_inducing)
    
    if args.fix_mean:
        filename += "_fixed_mean"
    if args.fix_variance:
        filename += "_fixed_variance"

    if args.fix_inducing:
        filename += "_fixed_inducing"

    if args.bb_alpha != 0:
        filename += "_alpha=" + str(args.bb_alpha)

    if args.split is not None:
        filename += "_split=" + str(args.split)

    filename += args.name_flag

    return filename
