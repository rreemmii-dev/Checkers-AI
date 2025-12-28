use crate::checkers::board::Board;
use crate::checkers::win_status::WinStatus::{Continue, Draw, Win};
use crate::players::alpha_beta::get_move::get_alpha_beta_move;
use crate::players::neural_network::get_move::{get_neural_network_move, input_of_board};
use crate::players::neural_network::neural_networks_types::neural_network::{
    NeuralNetworkTrait, export_parameters, generate_parameters, import_parameters,
};
use crate::{
    LEARNING_RATE_EVOLUTIONS, NB_LAYERS, NB_LEARNING_RATES, NB_LEARNINGS_PER_RESULT,
    NODES_PER_LAYER, NeuralNetwork,
};
use std::fs::read_dir;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const NB_NEURAL_NETWORKS: usize = 4;

#[derive(Clone)]
struct TrainingResult {
    neural_networks: Vec<NeuralNetwork>,
    nb_games: u64,
    nb_draws: u64,
}

fn train_neural_networks(
    neural_networks: Vec<NeuralNetwork>,
    duration: Duration,
) -> TrainingResult {
    let mut neural_networks = neural_networks;
    let t0 = Instant::now();

    let mut duels = Vec::new();
    for white in 0..NB_NEURAL_NETWORKS {
        for black in 0..NB_NEURAL_NETWORKS {
            if white == black {
                continue;
            }
            duels.push((white, black));
        }
    }

    let mut nb_iter = 0;
    let mut nb_games = 0;
    let mut nb_draws = 0;
    while t0.elapsed() < duration {
        // if nb_iter % 1 == 0 {
        //     println!("{nb_iter}, {:?}", t0.elapsed());
        // }
        nb_iter += 1;

        let mut handle = Vec::new();
        let neural_networks_copy = Arc::new(neural_networks.clone());
        for &(white, black) in &duels {
            let neural_networks_copy = neural_networks_copy.clone();
            handle.push(thread::spawn(move || {
                let mut board = Board::new();
                let mut boards_white = Vec::new();
                let mut boards_black = Vec::new();
                let mut white_plays = board.get_player_is_white();
                while !board.is_end_game() {
                    if white_plays {
                        boards_white.push(input_of_board(&board));
                        let m = get_neural_network_move(&board, &neural_networks_copy[white], true);
                        board.play(&m);
                    } else {
                        boards_black.push(input_of_board(&board));
                        let m = get_neural_network_move(&board, &neural_networks_copy[black], true);
                        board.play(&m);
                    }
                    white_plays = !white_plays;
                }

                let (white_result, black_result) = match board.get_win_status() {
                    Win(player) => {
                        if player.is_white() {
                            (1., 0.)
                        } else {
                            (0., 1.)
                        }
                    }
                    Draw => (0.5, 0.5),
                    Continue => panic!(),
                };

                let boards_played_white = boards_white
                    .into_iter()
                    .map(|b| (b, white_result))
                    .collect::<Vec<_>>();
                let boards_played_black = boards_black
                    .into_iter()
                    .map(|b| (b, black_result))
                    .collect::<Vec<_>>();

                let is_draw = board.get_win_status() == Draw;
                (
                    white,
                    black,
                    boards_played_white,
                    boards_played_black,
                    is_draw,
                )
            }))
        }
        for t in handle {
            let (white, black, boards_played_white, boards_played_black, is_draw) =
                t.join().unwrap();
            nb_games += 1;
            if is_draw {
                nb_draws += 1;
            }
            for _ in 0..NB_LEARNINGS_PER_RESULT {
                boards_played_white
                    .clone()
                    .into_iter()
                    .for_each(|(input, expected)| {
                        neural_networks[white].train_once(input, expected)
                    });
                boards_played_black
                    .clone()
                    .into_iter()
                    .for_each(|(input, expected)| {
                        neural_networks[black].train_once(input, expected)
                    });
            }
        }
    }
    TrainingResult {
        neural_networks,
        nb_games,
        nb_draws,
    }
}

fn train(folder: &str, duration: Duration) -> TrainingResult {
    let mut neural_networks = Vec::new();
    let mut files_in_folder = read_dir(folder).unwrap().peekable();
    if files_in_folder.peek().is_some() {
        for file in files_in_folder.map(Result::unwrap) {
            let new_neural_network =
                NeuralNetwork::import(import_parameters(file.path().to_str().unwrap()));
            neural_networks.push(new_neural_network);
        }
    } else {
        let mut nb_parameters = 0;
        for i in 0..NB_LAYERS - 1 {
            // Biases are neglected, we only count weights
            nb_parameters += NODES_PER_LAYER[i] * NODES_PER_LAYER[i + 1];
        }
        println!("**************************************");
        println!("*** Generating new neural networks ***");
        println!("***  {} parameters  ***", nb_parameters);
        println!("**************************************");
        for _ in 0..NB_NEURAL_NETWORKS {
            let new_neural_network =
                NeuralNetwork::import(generate_parameters(NODES_PER_LAYER, 0.000_1));
            neural_networks.push(new_neural_network);
        }
    }
    let mut handle = Vec::new();
    for &rate_evolution in LEARNING_RATE_EVOLUTIONS {
        let mut neural_networks = neural_networks.clone();
        neural_networks
            .iter_mut()
            .for_each(|nn| nn.change_learning_rate(rate_evolution));
        handle.push(thread::spawn(move || {
            train_neural_networks(neural_networks, duration)
        }));
    }
    let mut all_training_results = Vec::new();
    for t in handle {
        all_training_results.push(t.join().unwrap());
    }
    let mut scores = vec![0; NB_LEARNING_RATES];
    for white_team_id in 0..NB_LEARNING_RATES {
        for black_team_id in 0..NB_LEARNING_RATES {
            if white_team_id == black_team_id {
                continue;
            }
            // TODO: Multithreading this part
            for white_nn in 0..NB_NEURAL_NETWORKS {
                for black_nn in 0..NB_NEURAL_NETWORKS {
                    let mut board = Board::new();
                    let mut white_plays = board.get_player_is_white();
                    while !board.is_end_game() {
                        if white_plays {
                            board.play(&get_neural_network_move(
                                &board,
                                &all_training_results[white_team_id].neural_networks[white_nn],
                                true,
                            ));
                        } else {
                            board.play(&get_neural_network_move(
                                &board,
                                &all_training_results[black_team_id].neural_networks[black_nn],
                                true,
                            ));
                        }
                        white_plays = !white_plays;
                    }
                    match board.get_win_status() {
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
                        Continue => panic!(),
                    }
                }
            }
        }
    }
    let best_score = *scores.iter().max().unwrap();
    for (team_id, score) in scores.into_iter().enumerate() {
        if score == best_score {
            let training_result = all_training_results[team_id].clone();
            for (id, neural_network) in training_result.neural_networks.iter().enumerate() {
                export_parameters(
                    neural_network.export(),
                    &(folder.to_string() + "/neural_network_" + &id.to_string() + ".txt"),
                )
            }
            return training_result;
        }
    }
    panic!()
}

pub fn train_loop(folder: &str) {
    loop {
        // let training_result = train(folder, Duration::from_secs(1 * 60));
        let training_result = train(folder, Duration::from_secs(1 * 60 * 60));
        let nn = training_result.neural_networks[0].clone();
        let mut board = Board::new();
        while !board.is_end_game() {
            board.display();
            let vec = if board.get_player_is_white() {
                get_neural_network_move(&board, &nn, false)
            } else {
                get_alpha_beta_move(&board, false)
            };
            board.play(&vec);
        }
        board.display();
        println!("{:?}", board.get_win_status());
        println!("White: NN, Black: BFS");
        println!(
            "Training with: {} games, {} draws",
            training_result.nb_games, training_result.nb_draws
        );
    }
}
