use crate::checkers::board::Board;
use crate::checkers::win_status::WinStatus::{Continue, Draw, Win};
use crate::consts::{
    DEPTH_LIMIT, DEPTH_LIMIT_STRATEGY, NeuralNetwork, TIME_LIMIT_STRATEGY, TIME_PER_MOVE,
};
use crate::players::alpha_beta::get_move::{
    get_alpha_beta_move_depth_limit, get_alpha_beta_move_simple_heuristic_time_limit,
};
use crate::players::alpha_beta::score::naive_score;
use crate::players::neural_network::get_move::get_neural_network_move;
use std::sync::Arc;

pub fn compare_heuristics(neural_networks: &[NeuralNetwork]) {
    let (mut wins, mut draws, mut losses) = (0, 0, 0);
    for nn in neural_networks.iter().rev().take(100) {
        play(nn, true, true, &mut wins, &mut draws, &mut losses);
    }
    println!(
        "Plays white, Time limited - wins: {}, draws: {}, losses: {}",
        wins, draws, losses
    );

    let (mut wins, mut draws, mut losses) = (0, 0, 0);
    for nn in neural_networks.iter().rev().take(100) {
        play(nn, false, true, &mut wins, &mut draws, &mut losses);
    }
    println!(
        "Plays black, Time limited - wins: {}, draws: {}, losses: {}",
        wins, draws, losses
    );

    let (mut wins, mut draws, mut losses) = (0, 0, 0);
    for nn in neural_networks.iter().rev().take(100) {
        play(nn, true, false, &mut wins, &mut draws, &mut losses);
    }
    println!(
        "Plays white, Depth limited - wins: {}, draws: {}, losses: {}",
        wins, draws, losses
    );

    let (mut wins, mut draws, mut losses) = (0, 0, 0);
    for nn in neural_networks.iter().rev().take(100) {
        play(nn, false, false, &mut wins, &mut draws, &mut losses);
    }
    println!(
        "Plays black, Depth limited - wins: {}, draws: {}, losses: {}",
        wins, draws, losses
    );
}

fn play(
    neural_network: &NeuralNetwork,
    nn_plays_white: bool,
    is_time_limited: bool,
    wins: &mut u64,
    draws: &mut u64,
    losses: &mut u64,
) {
    let mut board = Board::new();
    let mut nn_plays = nn_plays_white;
    while !board.is_end_game() {
        let m = if nn_plays {
            if is_time_limited {
                get_neural_network_move(&board, neural_network, TIME_LIMIT_STRATEGY, true)
            } else {
                get_neural_network_move(&board, neural_network, DEPTH_LIMIT_STRATEGY, true)
            }
        } else {
            if is_time_limited {
                get_alpha_beta_move_simple_heuristic_time_limit(&board, TIME_PER_MOVE, true)
            } else {
                get_alpha_beta_move_depth_limit(&board, Arc::new(naive_score), DEPTH_LIMIT, true)
            }
        };
        board.play(&m);
        nn_plays = !nn_plays;
    }
    match board.get_win_status() {
        Win(player) => {
            if player.is_white() == nn_plays_white {
                *wins += 1;
            } else {
                *losses += 1;
            }
        }
        Draw => *draws += 1,
        Continue => panic!("Continue"),
    }
}
