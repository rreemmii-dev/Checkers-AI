use crate::checkers::win_status::WinStatus::Continue;
use crate::consts::{AI_DEPTH_LIMIT_STRATEGY_TOURNAMENT, NeuralNetwork};
use crate::neural_network::storage::load_all_neural_networks;
use crate::neural_network::training::graphs::display_results;
use crate::neural_network::training::heuristic_comparison::compare_heuristics;
use crate::neural_network::training::train::{TournamentResult, play_game};
#[cfg(not(nn_is_sync))]
use std::ops::Deref;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

pub fn run_tournament(folder: &str) {
    let t0 = Instant::now();
    let neural_networks = load_all_neural_networks(folder);
    println!("{} neural networks", neural_networks.len());
    println!("{:?}", t0.elapsed());
    let tournament_result = compute_tournament_result(&neural_networks);
    println!("{:?}", t0.elapsed());
    display_results(&tournament_result);
    compare_heuristics(&neural_networks);
}

fn compute_tournament_result(neural_networks: &[NeuralNetwork]) -> TournamentResult {
    let nb_neural_networks = neural_networks.len();
    let result = vec![vec![Continue; nb_neural_networks]; nb_neural_networks];
    let mut handle = Vec::new();
    let neural_networks = Arc::new(neural_networks.to_owned());
    let result = Arc::new(Mutex::new(result));
    for (index1, nn1) in neural_networks.iter().enumerate() {
        let neural_networks = cfg_select! {
            nn_is_sync => neural_networks.clone(),
            not(nn_is_sync) => neural_networks.deref().to_owned(),
        };
        let result = result.clone();
        let nn1 = nn1.to_owned();
        handle.push(thread::spawn(move || {
            for (index2, nn2) in neural_networks.iter().enumerate() {
                result.lock().unwrap()[index1][index2] =
                    play_game(&nn1, nn2, AI_DEPTH_LIMIT_STRATEGY_TOURNAMENT);
            }
        }));
    }
    for t in handle {
        t.join().unwrap();
    }
    result.lock().unwrap().to_owned()
}
