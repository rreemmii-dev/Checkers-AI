# Checkers-AI


A checkers (or draughts) game with an AI algorithm (for more information, see [#AI Algorithm](#ai-algorithm)), written in Rust.

Rules: [English draughts](https://en.wikipedia.org/wiki/English_draughts)

<div align="center">
A rather simple UI, running in a terminal:
<br>
<img src="https://rreemmii-dev.github.io/checkers-ai/screenshot.png" alt="Game screenshot" width="50%" />
</div>


## Table of content

- [Getting started](#getting-started)
- [AI Algorithm](#ai-algorithm)
- [Performance](#performance)
- [License](#license)


## Getting started

### Installation

```bash
# Clone the repository:
git clone git@github.com:rreemmii-dev/Checkers-AI.git

cd Checkers-AI
```

Ensure you have `cargo` installed

### Run

```bash
cargo run --release
```


## AI Algorithm

### General idea

The idea is to run several [negamax](https://en.wikipedia.org/wiki/Negamax) searches using [alpha–beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning), each in a different thread.

Thus, the AI runs in two parts:

1) A shallow multithreaded [depth-first search (DFS)](https://en.wikipedia.org/wiki/Depth-first_search), in which the AI goes 2 moves deep.

2) A [negamax](https://en.wikipedia.org/wiki/Negamax) search at the end of each branch of the initial DFS.

### Compute a board score

As the negamax exploration is run with a limited depth, a way to compute the score of a board is required.

First, a score has to be chosen for each piece. It depends on the piece type (man or king) and its position on the board. For exemple, men near promotion line and kings near the middle of the board are more valuable.

Then, a coefficient has to be chosen, representing how close is the game to be a draw (either by 3 repetitions, or by playing 2 * 40 moves without capture nor promotion). A coefficient of 50 reduces players' effective score by half. The following functions where chosen (rather arbitrarily):

<div align="center">
<img src="https://rreemmii-dev.github.io/checkers-ai/score_coef_no_capture.png" alt="No capture/promotion graph" width="45%" /> <img src="https://rreemmii-dev.github.io/checkers-ai/score_coef_repetitions.png" alt="Repetitions count graph" width="45%" />
</div>


## Performance

### Bitboards

Bitboards are a way to avoid iterating over each square, for exemple while searching for possible moves.

3 bitboards are enough to store the whole board: one for white pieces, one for black pieces and one for both white and black kings.

There is a total of 32 squares, each having a different index:

| 7  |    | 28 |    | 29 |    | 30 |    | 31 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 6  | 24 |    | 25 |    | 26 |    | 27 |    |
| 5  |    | 20 |    | 21 |    | 22 |    | 23 |
| 4  | 16 |    | 17 |    | 18 |    | 19 |    |
| 3  |    | 12 |    | 13 |    | 14 |    | 15 |
| 2  | 8  |    | 9  |    | 10 |    | 11 |    |
| 1  |    | 4  |    | 5  |    | 6  |    | 7  |
| 0  | 0  |    | 1  |    | 2  |    | 3  |    |
|    | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  |


## License

Distributed under the MIT License. See [LICENSE.md](LICENSE.md)
