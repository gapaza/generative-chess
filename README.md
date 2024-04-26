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






Evaluating opening...: 100%|██████████| 10/10 [00:02<00:00,  3.70it/s, result=0.327]
Evaluating middlegame...: 100%|██████████| 10/10 [00:00<00:00, 43.63it/s, result=0.281]
Evaluating endgame...: 100%|██████████| 10/10 [00:00<00:00, 44.48it/s, result=0.23]
Evaluating equality...: 100%|██████████| 9/9 [00:00<00:00, 14.80it/s, result=0.324]
Evaluating advantage...: 100%|██████████| 10/10 [00:00<00:00, 42.75it/s, result=0.299]
Evaluating mate...: 100%|██████████| 10/10 [00:00<00:00, 42.28it/s, result=0.167]
Evaluating fork...: 100%|██████████| 10/10 [00:00<00:00, 42.52it/s, result=0.235]
Evaluating pin...: 100%|██████████| 10/10 [00:00<00:00, 46.38it/s, result=0.239]








