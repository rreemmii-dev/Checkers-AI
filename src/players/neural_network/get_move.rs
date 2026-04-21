use crate::checkers::board::{
    BOARD_SIZE, Board, MAX_BOARD_COUNT, MAX_MOVES_WITHOUT_CAPTURE, Move, is_playable,
};
use crate::checkers::win_status::WinStatus::{Continue, Draw, Win};
use crate::consts::{NeuralNetwork, NeuralNetworkFloat};
use crate::neural_network::neural_network::NeuralNetworkTrait;
use crate::neural_network::types::matrix::Matrix;
use crate::players::alpha_beta::get_move::{
    get_alpha_beta_move_depth_limit, get_alpha_beta_move_time_limit,
};
use ChooseMoveStrategy::{DepthLimit, TimeLimit, Training};
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Copy)]
pub enum ChooseMoveStrategy {
    DepthLimit(i8),
    TimeLimit(Duration),
    Training,
}

pub fn get_neural_network_move(
    board: &Board,
    neural_network: &NeuralNetwork,
    move_strategy: ChooseMoveStrategy,
    threaded: bool,
) -> Move {
    let neural_network_clone = neural_network.clone();
    let heuristic = Arc::new(move |board: &Board| {
        // Current player POV
        let f64_score = neural_network_clone.get_output(input_of_board(board));
        ((f64_score - 0.5) * 1_000_000.) as i64
    });
    match move_strategy {
        DepthLimit(depth_limit) => {
            get_alpha_beta_move_depth_limit(board, heuristic, depth_limit, threaded)
        }
        TimeLimit(duration) => get_alpha_beta_move_time_limit(board, heuristic, duration, threaded),
        Training => {
            assert!(!threaded);
            get_neural_network_move_training(board, neural_network)
        }
    }
}

pub fn input_of_board(board: &Board) -> Matrix {
    let mut res = Matrix::zero(130, 1);
    let current_player = board.get_player_is_white();
    let mut i = 0;
    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            if is_playable(x, y) {
                if let Some(p) = board.get(x, y) {
                    let mut index = i;
                    if p.is_white() == current_player {
                        index += 2;
                    }
                    if p.is_king() {
                        index += 1;
                    }
                    res.set(index, 0, 1.);
                }
                i += 4;
            }
        }
    }
    res.set(
        128,
        0,
        board.get_board_count() as NeuralNetworkFloat / MAX_BOARD_COUNT as NeuralNetworkFloat,
    );
    res.set(
        129,
        0,
        board.get_moves_without_capture() as NeuralNetworkFloat
            / MAX_MOVES_WITHOUT_CAPTURE as NeuralNetworkFloat,
    );
    res
}

fn get_neural_network_move_training(board: &Board, neural_network: &NeuralNetwork) -> Move {
    let mut moves = Vec::new();
    let mut inputs = Vec::new();
    let mut fixed_scores = Vec::new();
    let self_is_white = board.get_player_is_white();
    for m in board.possible_moves() {
        let mut board = board.clone();
        board.play(&m);
        moves.push(m);
        inputs.push(input_of_board(&board));
        fixed_scores.push(match board.get_win_status() {
            Win(player) => {
                // Not 1. or 0.: needs to be unsigmoid-safe
                if player.is_white() == self_is_white {
                    Some(0.999)
                } else {
                    Some(0.001)
                }
            }
            Draw => Some(0.5),
            Continue => None,
        });
    }
    let mut scores = neural_network
        .get_outputs(inputs)
        .into_iter()
        .map(|opponent_score| 1. - opponent_score)
        .collect::<Vec<_>>();
    for (index, score_opt) in fixed_scores.into_iter().enumerate() {
        if let Some(score) = score_opt {
            scores[index] = score;
        }
    }

    choose_move_from_scores(scores, moves)
}

fn choose_move_from_scores(scores: Vec<NeuralNetworkFloat>, moves: Vec<Move>) -> Move {
    let mut scores = scores;
    for v in &mut scores {
        *v = NeuralNetworkFloat::exp(unsigmoid(*v));
    }
    let sum = scores.iter().sum::<NeuralNetworkFloat>();
    for v in &mut scores {
        *v /= sum;
    }
    let new_sum = scores.iter().sum::<NeuralNetworkFloat>();
    assert!((new_sum - 1.).abs() < 1e-5, "{}: {:?}", new_sum, scores);
    let val = rand::random::<NeuralNetworkFloat>();
    let mut current_sum = 0.;
    for (move_id, m) in moves.into_iter().enumerate() {
        current_sum += scores[move_id];
        if current_sum > val {
            return m;
        }
    }
    panic!("{} (target) > {} (sum of probabilities)", val, current_sum);
}

fn unsigmoid(y: NeuralNetworkFloat) -> NeuralNetworkFloat {
    NeuralNetworkFloat::ln(y / (1. - y))
}
