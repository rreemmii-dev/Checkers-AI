use crate::checkers::board::Board;
use crate::checkers::win_status::WinStatus;
use crate::checkers::win_status::WinStatus::{Continue, Draw, Win};
use crate::consts::{
    DEPTH_LIMIT_STRATEGY, LEARNING_RATE_EVOLUTIONS, NB_LEARNING_RATES, NB_LEARNINGS_PER_RESULT,
    NB_NEURAL_NETWORKS, NeuralNetwork, NeuralNetworkFloat, TIME_LIMIT_STRATEGY, TIME_PER_MOVE,
};
use crate::neural_network::neural_network::NeuralNetworkTrait;
use crate::neural_network::storage::{load_latest_neural_networks, store_new_neural_networks};
use crate::neural_network::types::matrix::Matrix;
use crate::players::alpha_beta::get_move::get_alpha_beta_move_simple_heuristic_time_limit;
use crate::players::neural_network::get_move::ChooseMoveStrategy::Training;
use crate::players::neural_network::get_move::{
    ChooseMoveStrategy, get_neural_network_move, input_of_board,
};
use chrono::Local;
#[cfg(not(nn_is_sync))]
use std::ops::Deref;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

pub type TournamentResult = Vec<Vec<WinStatus>>;

#[derive(Clone)]
struct TrainingResult {
    neural_networks: Vec<NeuralNetwork>,
    nb_games: u64,
    nb_draws: u64,
}

pub fn train_loop(folder: &str) {
    loop {
        let t0 = Instant::now();
        let training_result = train(folder, Duration::from_mins(5), 8);
        println!(
            "Training with: {} games, {} draws",
            training_result.nb_games, training_result.nb_draws
        );
        println!("Spent {:?}", t0.elapsed());
        let nn = training_result.neural_networks[0].clone();
        let mut board = Board::new();
        while !board.is_end_game() {
            println!("{}", board);
            let m = if board.get_player_is_white() {
                get_neural_network_move(&board, &nn, TIME_LIMIT_STRATEGY, true)
            } else {
                get_alpha_beta_move_simple_heuristic_time_limit(&board, TIME_PER_MOVE, true)
            };
            board.play(&m);
        }
        println!("{}", board);
        println!("{:?}", board.get_win_status());
        println!("White: NN, Black: BFS");
        println!(
            "Training with: {} games, {} draws",
            training_result.nb_games, training_result.nb_draws
        );
        println!("{:?} -> {}", t0.elapsed(), Local::now());
    }
}

pub fn play_game(
    white: &NeuralNetwork,
    black: &NeuralNetwork,
    move_strategy: ChooseMoveStrategy,
) -> WinStatus {
    let mut board = Board::new();
    let mut white_plays = board.get_player_is_white();
    while !board.is_end_game() {
        if white_plays {
            board.play(&get_neural_network_move(
                &board,
                white,
                move_strategy,
                false,
            ));
        } else {
            board.play(&get_neural_network_move(
                &board,
                black,
                move_strategy,
                false,
            ));
        }
        white_plays = !white_plays;
    }
    board.get_win_status()
}

fn train(folder: &str, duration_per_training: Duration, nb_trainings: usize) -> TrainingResult {
    let mut neural_networks = load_latest_neural_networks(folder);
    let mut nb_games = 0;
    let mut nb_draws = 0;
    for _ in 0..nb_trainings {
        let all_training_results =
            train_different_learning_rate(&neural_networks, duration_per_training);
        let training_result = keep_best_learning_rate(all_training_results);
        neural_networks = training_result.neural_networks;
        nb_games += training_result.nb_games;
        nb_draws += training_result.nb_draws;
    }
    store_new_neural_networks(&neural_networks, folder);
    TrainingResult {
        neural_networks,
        nb_games,
        nb_draws,
    }
}

fn train_different_learning_rate(
    neural_networks: &[NeuralNetwork],
    duration: Duration,
) -> Vec<TrainingResult> {
    let mut handle = Vec::new();
    for &rate_evolution in LEARNING_RATE_EVOLUTIONS {
        let mut neural_networks = neural_networks.to_owned();
        for nn in &mut neural_networks {
            nn.change_learning_rate(rate_evolution);
        }
        handle.push(thread::spawn(move || {
            train_neural_networks(neural_networks, duration)
        }));
    }
    handle
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect::<Vec<_>>()
}

fn keep_best_learning_rate(all_training_results: Vec<TrainingResult>) -> TrainingResult {
    let mut handle = Vec::new();
    let all_training_results = Arc::new(all_training_results);
    for white_team_id in 0..NB_LEARNING_RATES {
        for black_team_id in 0..NB_LEARNING_RATES {
            if white_team_id == black_team_id {
                continue;
            }
            let all_training_results = cfg_select! {
                nn_is_sync => all_training_results.clone(),
                not(nn_is_sync) => all_training_results.deref().to_owned(),
            };
            handle.push(thread::spawn(move || {
                let mut res = Vec::new();
                for white_nn in 0..NB_NEURAL_NETWORKS {
                    for black_nn in 0..NB_NEURAL_NETWORKS {
                        let game_result = play_game(
                            &all_training_results[white_team_id].neural_networks[white_nn],
                            &all_training_results[black_team_id].neural_networks[black_nn],
                            DEPTH_LIMIT_STRATEGY,
                        );
                        res.push((white_team_id, black_team_id, game_result));
                    }
                }
                res
            }));
        }
    }
    let mut scores = vec![0; NB_LEARNING_RATES];
    for t in handle {
        for (white_team_id, black_team_id, game_result) in t.join().unwrap() {
            match game_result {
                Win(player) => {
                    if player.is_white() {
                        scores[white_team_id] += 1;
                        scores[black_team_id] -= 1;
                    } else {
                        scores[black_team_id] += 1;
                        scores[white_team_id] -= 1;
                    }
                }
                Draw => (),
                Continue => panic!("Continue"),
            }
        }
    }

    let team_id = scores
        .into_iter()
        .enumerate()
        .max_by_key(|&(_index, score)| score)
        .unwrap()
        .0;
    all_training_results[team_id].clone()
}

fn train_neural_networks(
    neural_networks: Vec<NeuralNetwork>,
    duration: Duration,
) -> TrainingResult {
    let mut neural_networks = neural_networks;
    let t0 = Instant::now();
    let mut nb_games = 0;
    let mut nb_draws = 0;
    while t0.elapsed() < duration {
        let mut handle = Vec::new();
        let neural_networks_arc = Arc::new(neural_networks.clone());
        for (white, black) in get_duels_list() {
            let neural_networks = cfg_select! {
                nn_is_sync => neural_networks_arc.clone(),
                not(nn_is_sync) => neural_networks_arc.deref().to_owned(),
            };
            handle.push(thread::spawn(move || {
                let (boards_played_white, boards_played_black, win_status) =
                    play_game_return_boards(&neural_networks[white], &neural_networks[black]);

                let (white_result, black_result) = get_score_from_win_status(win_status);

                let is_draw = win_status == Draw;
                (
                    white,
                    black,
                    boards_played_white,
                    boards_played_black,
                    white_result,
                    black_result,
                    is_draw, // Only used to print debug data
                )
            }));
        }
        let mut results = vec![Vec::new(); neural_networks.len()];
        for t in handle {
            let (
                white,
                black,
                boards_played_white,
                boards_played_black,
                white_result,
                black_result,
                is_draw,
            ) = t.join().unwrap();
            nb_games += 1;
            if is_draw {
                nb_draws += 1;
            }
            results[white].push((boards_played_white, white_result));
            results[black].push((boards_played_black, black_result));
        }
        neural_networks = train_from_results(&neural_networks, &results);
    }
    TrainingResult {
        neural_networks,
        nb_games,
        nb_draws,
    }
}

fn play_game_return_boards(
    nn_white: &NeuralNetwork,
    nn_black: &NeuralNetwork,
) -> (Vec<Matrix>, Vec<Matrix>, WinStatus) {
    let mut board = Board::new();
    let mut boards_played_white = Vec::new();
    let mut boards_played_black = Vec::new();
    let mut white_plays = board.get_player_is_white();
    while !board.is_end_game() {
        if white_plays {
            boards_played_white.push(input_of_board(&board));
            let m = get_neural_network_move(&board, nn_white, Training, false);
            board.play(&m);
        } else {
            boards_played_black.push(input_of_board(&board));
            let m = get_neural_network_move(&board, nn_black, Training, false);
            board.play(&m);
        }
        white_plays = !white_plays;
    }
    (
        boards_played_white,
        boards_played_black,
        board.get_win_status(),
    )
}

fn get_score_from_win_status(win_status: WinStatus) -> (NeuralNetworkFloat, NeuralNetworkFloat) {
    match win_status {
        Win(player) => {
            if player.is_white() {
                (1., 0.)
            } else {
                (0., 1.)
            }
        }
        Draw => (0.5, 0.5),
        Continue => panic!("Continue"),
    }
}

fn train_from_results(
    neural_networks: &[NeuralNetwork],
    results: &[Vec<(Vec<Matrix>, NeuralNetworkFloat)>],
) -> Vec<NeuralNetwork> {
    let mut handle = Vec::new();
    for i in 0..neural_networks.len() {
        let mut neural_network = neural_networks[i].clone();
        let results = results[i].clone();
        handle.push(thread::spawn(move || {
            for (boards_played, result) in &results {
                train_from_result(&mut neural_network, boards_played, *result);
            }
            neural_network
        }));
    }
    handle
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect::<Vec<_>>()
}

fn train_from_result(
    neural_network: &mut NeuralNetwork,
    boards_played: &[Matrix],
    score: NeuralNetworkFloat,
) {
    for _ in 0..NB_LEARNINGS_PER_RESULT {
        for input in boards_played {
            neural_network.train_once(input, score);
        }
    }
}

fn get_duels_list() -> Vec<(usize, usize)> {
    let mut duels = Vec::new();
    for white in 0..NB_NEURAL_NETWORKS {
        for black in 0..NB_NEURAL_NETWORKS {
            if white == black {
                continue;
            }
            duels.push((white, black));
        }
    }
    duels
}
