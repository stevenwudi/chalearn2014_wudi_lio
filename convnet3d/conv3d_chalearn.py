class conn3d_chalearn(object):
    """
    this is the class for generating chalearn convnet 3d moduele
    """

    for c in (use, lr, batch, net, reg, drop, mom, tr):
    write(c.__name__+":", res_dir)
    _s = c.__dict__
    del _s['__module__'], _s['__doc__']
    for key in _s.keys(): 
        val = str(_s[key])
        if val.startswith("<static"): val = str(_s[key].__func__.__name__)
        if val.startswith("<Cuda"): continue
        if val.startswith("<Tensor"): continue
        write("  "+key+": "+val, res_dir)


    ####################################################################
    ####################################################################
    print "\n%s\n\tbuilding\n%s"%(('-'*30,)*2)
    ####################################################################
    #################################################################### 
    # ConvNet
    # ------------------------------------------------------------------------------
    # calculate resulting video shapes for all stages
    conv_shapes = []
    for i in xrange(net.n_stages):
        k,p,v = array(net.kernels[i]), array(net.pools[i]), array(tr.video_shapes[i])
        conv_s = tuple(v-k+1)
        conv_shapes.append(conv_s)
        tr.video_shapes.append(tuple((v-k+1)/p))
        print "stage", i
        if use.depth and i==0:
            print "  conv",tr.video_shapes[i],"x 2 ->",conv_s #for body and hand
        else:
            print "  conv",tr.video_shapes[i],"->",conv_s
        print "  pool",conv_s,"->",tr.video_shapes[i+1],"x",net.maps[i+1]

    # number of inputs for MLP = (# maps last stage)*(# convnets)*(resulting video shape) + trajectory size
    n_in_MLP = net.maps[-1]*net.n_convnets*prod(tr.video_shapes[-1]) 
    print 'MLP:', n_in_MLP, "->", net.hidden, "->", net.n_class, ""

    if use.depth:
        if net.n_convnets==2: 
            out = [x[:,:,0,:,:,:], x[:,:,1,:,:,:]] # 2 nets: body and hand

    # build 3D ConvNet
    insp = []
    for stage in xrange(net.n_stages):
        for i in xrange(len(out)): # for body and hand
            # normalization
            if use.norm and stage==0: 
                gray_norm = NormLayer(out[i][:,0:1], method="lcn",
                    use_divisor=use.norm_div).output
                gray_norm = std_norm(gray_norm,axis=[-3,-2,-1])
                depth_norm = var_norm(out[i][:,1:])
                out[i]  = T.concatenate([gray_norm,depth_norm],axis=1)
            elif use.norm:
                out[i] = NormLayer(out[i], method="lcn",use_divisor=use.norm_div).output
                out[i] = std_norm(out[i],axis=[-3,-2,-1])
            # convolutions  
            out[i] *= net.scaler[stage][i]
            layers.append(ConvLayer(out[i], **conv_args(stage, i, batch, net, use, tr.rng, tr.video_shapes)))
            out[i] = layers[-1].output
            out[i] = PoolLayer(out[i], net.pools[stage], method=net.pool_method).output
            if tr.inspect: insp.append(T.mean(out[i]))

    # flatten all convnets outputs
    for i in xrange(len(out)): out[i] = std_norm(out[i],axis=[-3,-2,-1])
    out = [out[i].flatten(2) for i in range(len(out))]
    vid_ = T.concatenate(out, axis=1)

    # dropout
    if use.drop: 
        vid_ = DropoutLayer(vid_, rng=tr.rng, p=drop.p_vid).output

    #maxout
    if use.maxout:
        vid_ = maxout(vid_, (batch.micro,n_in_MLP))
        net.activation = lin
        n_in_MLP /= 2
        # net.hidden *= 2

    # MLP
    # ------------------------------------------------------------------------------
    # fusion
    if net.fusion == "early":
        out = vid_
        # hidden layer
        layers.append(HiddenLayer(out, n_in=n_in_MLP, n_out=net.hidden, rng=tr.rng, 
            W_scale=net.W_scale[-2], b_scale=net.b_scale[-2], activation=relu))
        out = layers[-1].output


    if tr.inspect: insp = T.stack(insp[0],insp[1],insp[2],insp[3],insp[4],insp[5], T.mean(out))
    else: insp =  T.stack(0,0)
    # out = normalize(out)
    if use.drop: out = DropoutLayer(out, rng=tr.rng, p=drop.p_hidden).output
    #maxout
    if use.maxout:
        out = maxout(out, (batch.micro,net.hidden))
        net.hidden /= 2

