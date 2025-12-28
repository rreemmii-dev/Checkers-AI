use crate::checkers::board::{
    BOARD_SIZE, Board, MAX_BOARD_COUNT, MAX_MOVES_WITHOUT_CAPTURE, is_playable,
};
use crate::players::alpha_beta::alpha_beta::threaded_moves_list;
use crate::players::neural_network::neural_networks_types::matrix::Matrix;
use crate::players::neural_network::neural_networks_types::neural_network::NeuralNetworkTrait;
use crate::{
    AI_TIME_PER_MOVE, BEST_MOVE_FIRST_MIN_DEPTH, BEST_MOVE_FIRST_SKIP_SIZE, MAX_DEPTH_TRAINING,
    MAX_THREADS_DEPTH, NeuralNetwork,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::thread::sleep;

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
        board.get_board_count() as f64 / MAX_BOARD_COUNT as f64,
    );
    res.set(
        129,
        0,
        board.get_moves_without_capture() as f64 / MAX_MOVES_WITHOUT_CAPTURE as f64,
    );
    res
}

fn get_neural_network_move_old(
    board: &Board,
    neural_network: &NeuralNetwork,
    is_training: bool,
) -> Vec<(i8, i8)> {
    // TODO: Really better than the regular one?
    let arc_nn = Arc::new(neural_network.to_owned());

    if board.is_end_game() {
        return Vec::new();
    }

    let best_moves = if is_training {
        let heuristic_nn = move |board: &Board| {
            // Current player POV
            let f64_score = arc_nn.get_output(input_of_board(board));
            ((f64_score - 0.5) * 1_000_000.) as i64
        };
        // alpha_beta_list(
        //     &board,
        //     &heuristic_nn,
        //     BEST_MOVE_FIRST_MIN_DEPTH,
        //     BEST_MOVE_FIRST_SKIP_SIZE,
        //     MAX_DEPTH_TRAINING,
        //     &AtomicBool::new(false),
        // )
        threaded_moves_list(
            board.clone(),
            Arc::new(heuristic_nn),
            BEST_MOVE_FIRST_MIN_DEPTH,
            BEST_MOVE_FIRST_SKIP_SIZE,
            MAX_DEPTH_TRAINING,
            MAX_THREADS_DEPTH,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap()
    } else {
        let mut best_moves = Vec::new();
        let mut depth = 2 * 1;
        let cancel_search = Arc::new(AtomicBool::new(false));
        {
            let cancel_search = cancel_search.clone();
            thread::spawn(move || {
                sleep(AI_TIME_PER_MOVE);
                cancel_search.store(true, Ordering::Release);
            });
        }
        while !cancel_search.load(Ordering::Acquire) {
            let arc_nn = arc_nn.clone();
            let heuristic_nn = move |board: &Board| {
                // Current player POV
                let f64_score = arc_nn.get_output(input_of_board(board));
                ((f64_score - 0.5) * 1_000_000.) as i64
            };

            if let Some(new_best_moves) = threaded_moves_list(
                board.clone(),
                Arc::new(heuristic_nn),
                BEST_MOVE_FIRST_MIN_DEPTH,
                BEST_MOVE_FIRST_SKIP_SIZE,
                depth,
                MAX_THREADS_DEPTH,
                cancel_search.clone(),
            ) {
                // if let Some(new_best_moves) = alpha_beta_list(
                //     &board,
                //     &heuristic_nn,
                //     BEST_MOVE_FIRST_MIN_DEPTH,
                //     BEST_MOVE_FIRST_SKIP_SIZE,
                //     depth,
                //     &cancel_search,
                // ) {
                best_moves = new_best_moves;
                depth += 2;
                if depth >= 2 * 50 {
                    // Probably end of game, or only 1 move allowed
                    break;
                }
            }
        }
        best_moves
    };

    let x = rand::random_range(0..best_moves.len());
    best_moves[x].clone()
}

pub fn get_neural_network_move(
    board: &Board,
    neural_network: &NeuralNetwork,
    is_training: bool,
) -> Vec<(i8, i8)> {
    if !is_training {
        return get_neural_network_move_old(board, neural_network, is_training);
    }
    let mut moves = Vec::new();
    let mut inputs = Vec::new();
    for m in board.possible_moves() {
        let mut board = board.clone();
        board.play(&m);
        let input = input_of_board(&board);
        moves.push(m);
        inputs.push(input);
    }
    let mut output = neural_network
        .get_outputs(inputs)
        .into_iter()
        .map(|res| -res)
        .collect::<Vec<_>>();
    // TODO: Cf how to choose the best move according to scores?
    let sum = output.iter().map(|x| x.abs()).sum::<f64>();
    output.iter_mut().for_each(|v| *v /= sum);
    output.iter_mut().for_each(|v| *v = v.exp());
    let sum = output.iter().sum::<f64>();
    output.iter_mut().for_each(|v| *v /= sum);
    assert!((output.iter().sum::<f64>() - 1.).abs() < 1e-10);
    let val = rand::random::<f64>();
    let mut current_sum = 0.;
    for (move_id, m) in moves.into_iter().enumerate() {
        let v = output[move_id];
        current_sum += v;
        if current_sum > val {
            return m;
        }
    }
    println!("{}", current_sum);
    panic!();
}
