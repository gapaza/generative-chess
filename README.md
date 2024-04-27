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






Evaluating opening...: 100%|██████████| 10/10 [00:03<00:00,  2.87it/s, result=0.398]
Evaluating middlegame...: 100%|██████████| 10/10 [00:00<00:00, 16.87it/s, result=0.345]
Evaluating endgame...: 100%|██████████| 10/10 [00:00<00:00, 23.84it/s, result=0.315]
Evaluating equality...: 100%|██████████| 9/9 [00:01<00:00,  8.81it/s, result=0.397]
Evaluating advantage...: 100%|██████████| 10/10 [00:00<00:00, 25.99it/s, result=0.398]
Evaluating mate...: 100%|██████████| 10/10 [00:00<00:00, 26.39it/s, result=0.219]
Evaluating fork...: 100%|██████████| 10/10 [00:00<00:00, 25.52it/s, result=0.301]
Evaluating pin...: 100%|██████████| 10/10 [00:00<00:00, 25.25it/s, result=0.308]



Evaluating opening...: 100%|██████████| 10/10 [00:03<00:00,  3.00it/s, result=0.42]
Evaluating middlegame...: 100%|██████████| 10/10 [00:00<00:00, 26.46it/s, result=0.381]
Evaluating endgame...: 100%|██████████| 10/10 [00:00<00:00, 25.95it/s, result=0.339]
Evaluating equality...: 100%|██████████| 9/9 [00:01<00:00,  8.90it/s, result=0.431]
Evaluating advantage...: 100%|██████████| 10/10 [00:00<00:00, 25.08it/s, result=0.452]
Evaluating mate...: 100%|██████████| 10/10 [00:00<00:00, 25.19it/s, result=0.257]
Evaluating fork...: 100%|██████████| 10/10 [00:00<00:00, 25.53it/s, result=0.339]
Evaluating pin...: 100%|██████████| 10/10 [00:00<00:00, 25.89it/s, result=0.343]


Evaluating opening...: 100%|██████████| 10/10 [00:03<00:00,  2.99it/s, result=0.461]
Evaluating middlegame...: 100%|██████████| 10/10 [00:00<00:00, 25.85it/s, result=0.417]
Evaluating endgame...: 100%|██████████| 10/10 [00:00<00:00, 25.96it/s, result=0.391]
Evaluating equality...: 100%|██████████| 9/9 [00:01<00:00,  8.91it/s, result=0.484]
Evaluating advantage...: 100%|██████████| 10/10 [00:00<00:00, 25.82it/s, result=0.473]
Evaluating mate...: 100%|██████████| 10/10 [00:00<00:00, 25.65it/s, result=0.28]
Evaluating fork...: 100%|██████████| 10/10 [00:00<00:00, 23.48it/s, result=0.387]
Evaluating pin...: 100%|██████████| 10/10 [00:00<00:00, 25.76it/s, result=0.374]


Evaluating opening...: 100%|██████████| 10/10 [00:03<00:00,  2.68it/s, result=0.49]
Evaluating middlegame...: 100%|██████████| 10/10 [00:00<00:00, 24.76it/s, result=0.432]
Evaluating endgame...: 100%|██████████| 10/10 [00:00<00:00, 24.34it/s, result=0.424]
Evaluating equality...: 100%|██████████| 9/9 [00:01<00:00,  7.81it/s, result=0.508]
Evaluating advantage...: 100%|██████████| 10/10 [00:00<00:00, 24.34it/s, result=0.496]
Evaluating mate...: 100%|██████████| 10/10 [00:00<00:00, 24.26it/s, result=0.284]
Evaluating fork...: 100%|██████████| 10/10 [00:00<00:00, 24.75it/s, result=0.424]
Evaluating pin...: 100%|██████████| 10/10 [00:00<00:00, 23.98it/s, result=0.407]



Evaluating opening...: 100%|██████████| 10/10 [00:03<00:00,  2.94it/s, result=0.539]
Evaluating middlegame...: 100%|██████████| 10/10 [00:00<00:00, 25.17it/s, result=0.485]
Evaluating endgame...: 100%|██████████| 10/10 [00:00<00:00, 25.15it/s, result=0.474]
Evaluating equality...: 100%|██████████| 9/9 [00:01<00:00,  8.92it/s, result=0.557]
Evaluating advantage...: 100%|██████████| 10/10 [00:00<00:00, 25.59it/s, result=0.537]
Evaluating mate...: 100%|██████████| 10/10 [00:00<00:00, 25.92it/s, result=0.326]
Evaluating fork...: 100%|██████████| 10/10 [00:00<00:00, 26.06it/s, result=0.485]
Evaluating pin...: 100%|██████████| 10/10 [00:00<00:00, 25.83it/s, result=0.455]


Evaluating opening...: 100%|██████████| 10/10 [00:03<00:00,  3.02it/s, result=0.564]
Evaluating middlegame...: 100%|██████████| 10/10 [00:00<00:00, 26.57it/s, result=0.487]
Evaluating endgame...: 100%|██████████| 10/10 [00:00<00:00, 26.82it/s, result=0.501]
Evaluating equality...: 100%|██████████| 9/9 [00:00<00:00,  9.04it/s, result=0.592]
Evaluating advantage...: 100%|██████████| 10/10 [00:00<00:00, 26.93it/s, result=0.54]
Evaluating mate...: 100%|██████████| 10/10 [00:00<00:00, 27.34it/s, result=0.346]
Evaluating fork...: 100%|██████████| 10/10 [00:00<00:00, 26.40it/s, result=0.496]
Evaluating pin...: 100%|██████████| 10/10 [00:00<00:00, 26.40it/s, result=0.456]_



Evaluating opening...: 100%|██████████| 10/10 [00:03<00:00,  2.96it/s, result=0.561]
Evaluating middlegame...: 100%|██████████| 10/10 [00:00<00:00, 25.11it/s, result=0.5]
Evaluating endgame...: 100%|██████████| 10/10 [00:00<00:00, 24.94it/s, result=0.501]
Evaluating equality...: 100%|██████████| 9/9 [00:01<00:00,  8.81it/s, result=0.6]
Evaluating advantage...: 100%|██████████| 10/10 [00:00<00:00, 25.25it/s, result=0.547]
Evaluating mate...: 100%|██████████| 10/10 [00:00<00:00, 25.13it/s, result=0.37]
Evaluating fork...: 100%|██████████| 10/10 [00:00<00:00, 25.30it/s, result=0.519]
Evaluating pin...: 100%|██████████| 10/10 [00:00<00:00, 26.24it/s, result=0.472]    









