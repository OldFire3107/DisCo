import torch

def distance_corr(var_1,var_2,normedweight=None,power=2,exponent=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Overall power of the distance correlation. Default is 2, 1 gives the standard dcorr package
    exponent: Power of the distance. Default is 1, should be between (0,2) to decorrelate
    
    va1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """
    
    if normedweight is None:
        normedweight = torch.ones(var_1.shape[0], device=var_1.device)

    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    if exponent == 1:
        amat = (xx-yy).abs()
    else:
        amat = (xx-yy).abs()**exponent

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    if exponent == 1:
        bmat = (xx-yy).abs()
    else:
        bmat = (xx-yy).abs()**exponent

    amatavg = torch.mean(amat*normedweight,dim=1)
    amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)
    
    ABavg = torch.mean(amat*bmat*normedweight,dim=1)
    AAavg = torch.mean(amat*amat*normedweight,dim=1)
    BBavg = torch.mean(bmat*bmat*normedweight,dim=1)

    if power == 2:
        dCorr=torch.mean(ABavg*normedweight)/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif power == 1:
        dCorr=torch.sqrt(torch.mean(ABavg*normedweight)/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))
    else:
        dCorr=torch.mean(ABavg*normedweight)/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))**(power/2)
    
    return dCorr

    # xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    # yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    # if exponent == 1:
    #     amat = (xx-yy).abs()
    # else:
    #     amat = (xx-yy).abs()**exponent

    # xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    # yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    # if exponent == 1:
    #     bmat = (xx-yy).abs()
    # else:
    #     bmat = (xx-yy).abs()**exponent

    # # amatavg = torch.mean(amat*normedweight,dim=1)
    # # amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
    # #     -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
    # #     +torch.mean(amatavg*normedweight)

    # # bmatavg = torch.mean(bmat*normedweight,dim=1)
    # # bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
    # #     -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
    # #     +torch.mean(bmatavg*normedweight)
    
    # # ABavg = torch.mean(amat*bmat*normedweight,dim=1)
    # # AAavg = torch.mean(amat*amat*normedweight,dim=1)
    # # BBavg = torch.mean(bmat*bmat*normedweight,dim=1)


    # amatavg = torch.mean(amat*normedweight,dim=1)
    # amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
    #     -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
    #     +torch.mean(amatavg*normedweight)

    # bmatavg = torch.mean(bmat*normedweight,dim=1)
    # bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
    #     -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
    #     +torch.mean(bmatavg*normedweight)
    
    # ABavg = torch.mean(amat*bmat*normedweight,dim=1)
    # AAavg = torch.mean(amat*amat,dim=1)
    # BBavg = torch.mean(bmat*bmat,dim=1)

    # ABavg = torch.mean(ABavg*normedweight)
    # AAavg = torch.mean(AAavg)
    # BBavg = torch.mean(BBavg)

    # if power == 2:
    #     dCorr=ABavg/torch.sqrt(AAavg*BBavg)
    # elif power == 1:
    #     dCorr=torch.sqrt(ABavg/torch.sqrt(AAavg*BBavg))
    # else:
    #     dCorr=(ABavg/torch.sqrt(AAavg*BBavg))**(power/2)
    
    # return dCorr

import torch

def distance_corr_unbiased(var_1,var_2,normedweight=None,power=2,exponent=1):
    """var_1: First variable to decorrelate (eg mass)
    var_2: Second variable to decorrelate (eg classifier output)
    normedweight: Per-example weight. Sum of weights should add up to N (where N is the number of examples)
    power: Overall power of the distance correlation. Default is 2, 1 gives the standard dcorr package
    exponent: Power of the distance. Default is 1, should be between (0,2) to decorrelate
    
    va1_1, var_2 and normedweight should all be 1D torch tensors with the same number of entries
    
    Usage: Add to your loss function. total_loss = BCE_loss + lambda * distance_corr
    """

    N = var_1.shape[0]
    
    if normedweight is None:
        normedweight = torch.ones(var_1.shape[0], device=var_1.device)

    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    if exponent == 1:
        amat = (xx-yy).abs()
    else:
        amat = (xx-yy).abs()**exponent

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    if exponent == 1:
        bmat = (xx-yy).abs()
    else:
        bmat = (xx-yy).abs()**exponent

    amatavg = torch.mean(amat*normedweight,dim=1) * (N/(N-2))
    amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight) * (N/(N-1))

    bmatavg = torch.mean(bmat*normedweight,dim=1) * (N/(N-2))
    bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight) * (N/(N-1))
    
    ABavg = torch.mean(amat*bmat*normedweight,dim=1)
    AAavg = torch.mean(amat*amat*normedweight,dim=1)
    BBavg = torch.mean(bmat*bmat*normedweight,dim=1)
    
    ABavg = torch.mean(ABavg*normedweight) * (N/(N-3))
    AAavg = torch.mean(AAavg*normedweight) * (N/(N-3))
    BBavg = torch.mean(BBavg*normedweight) * (N/(N-3))

    if power == 2:
        dCorr=ABavg/torch.sqrt(AAavg*BBavg)
    elif power == 1:
        dCorr=torch.sqrt(ABavg/torch.sqrt(AAavg*BBavg))
    else:
        dCorr=(ABavg/torch.sqrt(AAavg*BBavg))**(power/2)
    
    return dCorr
    