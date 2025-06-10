# Generative Chess

This project exists on two servers
1. An aws p3.2xlarge instance for pre-training
2. An aws ri7z.8xlarge instance for fine-tuning (cpu bound updates)




### ChessGPT-v6-puzzle
opening    0.749, 0.754, 0.756, 0.762, 0.782
middlegame 0.652, 0.655, 0.663, 0.666, 0.681
endgame    0.663, 0.669, 0.673, 0.675, 0.685
equality   0.584, 0.589, 0.593, 0.596, 0.626
advantage  0.610, 0.613, 0.616, 0.618, 0.622
mate       0.764, 0.769, 0.779, 0.786, 0.808
fork       0.653, 0.661, 0.670, 0.673, 0.695
pin        0.593, 0.600, 0.606, 0.609, 0.672


### ChessGPT-v6
opening    0.721
middlegame 0.621
endgame    0.637
equality   0.572
advantage  0.589
mate       0.721
fork       0.634
pin        0.573


### ChessGPT-s
opening    0.62
middlegame 0.536
endgame    0.57
equality   0.53
advantage  0.538
mate       0.59
fork       0.551
pin        0.485


### ChessGPT-m
opening    0.638
middlegame 0.557
endgame    0.58
equality   0.567
advantage  0.546
mate       0.609
fork       0.572
pin        0.497






SelfPlay 3001 - 0  ret1: 0.1437 | ret2: 0.1584 | gamelen: 65.1016 | chkm1: 6.0000 | chkm2: 9.0000 | ill1: 55.0000 | ill2: 53.0000 | ent1: 2.8258 | ent2: 2.7592 | kl1: 0.0016 | kl2: 0.0017 | l1: -0.0006 | l2: -0.0005 | c_loss 0.37611523 | model updates 21
SelfPlay 3001 - 1  ret1: 0.1283 | ret2: 0.1550 | gamelen: 65.0078 | chkm1: 2.0000 | chkm2: 8.0000 | ill1: 56.0000 | ill2: 57.0000 | ent1: 2.7774 | ent2: 2.7670 | kl1: 0.0016 | kl2: 0.0017 | l1: -0.0005 | l2: -0.0004 | c_loss 0.30585465 | model updates 34
SelfPlay 3001 - 2  ret1: 0.1531 | ret2: 0.1502 | gamelen: 70.2734 | chkm1: 6.0000 | chkm2: 5.0000 | ill1: 49.0000 | ill2: 62.0000 | ent1: 2.7345 | ent2: 2.7321 | kl1: 0.0015 | kl2: 0.0020 | l1: -0.0005 | l2: -0.0005 | c_loss 0.26986605 | model updates 46
SelfPlay 3001 - 3  ret1: 0.1612 | ret2: 0.1451 | gamelen: 64.5234 | chkm1: 11.0000 | chkm2: 7.0000 | ill1: 44.0000 | ill2: 61.0000 | ent1: 2.8327 | ent2: 2.7645 | kl1: 0.0017 | kl2: 0.0016 | l1: -0.0005 | l2: -0.0003 | c_loss 0.1961193 | model updates 57
SelfPlay 3001 - 4  ret1: 0.1293 | ret2: 0.1344 | gamelen: 64.5547 | chkm1: 2.0000 | chkm2: 4.0000 | ill1: 56.0000 | ill2: 63.0000 | ent1: 2.8394 | ent2: 2.7835 | kl1: 0.0017 | kl2: 0.0016 | l1: -0.0005 | l2: -0.0005 | c_loss 0.159336 | model updates 66
SelfPlay 3001 - 5  ret1: 0.1354 | ret2: 0.1387 | gamelen: 61.0547 | chkm1: 6.0000 | chkm2: 7.0000 | ill1: 62.0000 | ill2: 50.0000 | ent1: 2.8353 | ent2: 2.8646 | kl1: 0.0017 | kl2: 0.0020 | l1: -0.0004 | l2: -0.0006 | c_loss 0.121056914 | model updates 76