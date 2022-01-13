def create_file_name(args):
    filename = ""

    filename += "dataset=" + args.dataset_name 
    filename += "_vip_layers=" + "-".join(str(i) for i in args.vip_layers) 
    filename += "_epochs=" + str(args.epochs) 
    filename += "_dropout=" + str(args.dropout) 
    filename += "_lr=" + str(args.lr) 
    
    if args.genf == "BNN":
        dims = "-".join(str(i) for i in args.bnn_structure)
        filename += "_genf=BNN_bnn-structure=" + dims 
    else:
        filename += "_genf=BNN-GP_inner-dim=" + str(args.bnn_inner_dim)  
        
    filename += "_alpha=" + str(args.bb_alpha)
    
    if args.prior_kl:
        filename += "_prior_kl"
    if args.zero_mean_prior:
        filename += "_zero_mean_prior"
    if args.fix_prior_noise:
        filename += "_prior_fixed_noise" 

    filename += "_split=" + str(args.split)

    return filename
