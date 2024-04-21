# Pretraining Experiment 1



## Introduction

The purpose of this experiment is to explore how scaling the size of ChessGPT (v1) impacts performance on a set of 
evaluation tasks (evals) derived from chess puzzles. The evals are as follows:
- Openings
- Middlegame
- Endgame
- Equality
- Advantage
- Checkmate
- Pin
- Fork



## Models

Three models are trained in this experiment:
- ChessGPT-S with 27m parameters
  - 8 decoders
    - 8 heads
    - 2048 dense
    - 512 embedding
- ChessGPT-M with m parameters
  - 16 decoders
    - 16 heads
    - 2048 dense
    - 1024 embedding
- ChessGPT-L with m parameters
  - 32 decoders
    - 32 heads
    - 2048 dense
    - 1024 embedding




## Training


### Dataset

The dataset consists of 10 million leela 0 games, and 7 million GM games, and 2 million puzzle-based games.





















