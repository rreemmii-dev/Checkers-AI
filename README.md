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

First, a score has to be chosen for each piece:
- 1 point for a man
- 2 points for a king

Then, a coefficient has to be chosen, representing how close is the game to be a draw (either by 3 repetitions, or by playing 2 * 40 moves without capture nor promotion). A coefficient of 50 reduces players' effective score by half. The following functions where chosen (rather arbitrarily):

<div align="center">
<img src="https://rreemmii-dev.github.io/checkers-ai/score_coef_no_capture.png" alt="No capture/promotion graph" width="45%" /> <img src="https://rreemmii-dev.github.io/checkers-ai/score_coef_repetitions.png" alt="Repetitions count graph" width="45%" />
</div>


## License

Distributed under the MIT License. See [LICENSE.md](LICENSE.md)
