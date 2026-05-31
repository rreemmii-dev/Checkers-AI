use crate::consts::Mode::{Play, Tournament, Train};
use crate::neural_network;
use crate::players::neural_network::get_move::ChooseMoveStrategy;
use crate::players::neural_network::get_move::ChooseMoveStrategy::{DepthLimit, TimeLimit};
use std::time::Duration;

#[derive(Eq, PartialEq)]
pub enum Mode {
    Play,
    Train,
    Tournament,
}

/// Default time spent by the AI to choose its next move
pub const TIME_PER_MOVE: Duration = Duration::from_secs(1);
/// Choose move strategy using time limit
pub const TIME_LIMIT_STRATEGY: ChooseMoveStrategy = TimeLimit(TIME_PER_MOVE);
/// Default alpha beta exploration depth to choose the AI next move
pub const DEPTH_LIMIT: i8 = 4;
/// Choose move strategy using depth limit
pub const DEPTH_LIMIT_STRATEGY: ChooseMoveStrategy = DepthLimit(DEPTH_LIMIT);

/// The default number of parameters in each layer of the neural network, for newly created neural networks only
// pub const DEFAULT_NODES_PER_LAYER: [usize; 4] = [130, 512, 256, 1];
pub const DEFAULT_NODES_PER_LAYER: [usize; 5] = [130, 1024, 256, 128, 1];
// pub const DEFAULT_NODES_PER_LAYER: [usize; 6] = [130, 1024, 512, 256, 128, 1];
// pub const DEFAULT_NODES_PER_LAYER: [usize; 8] = [130, 512, 1024, 1024, 512, 256, 128, 1];
/// The default number of neural network layers, for newly created neural networks only
pub const DEFAULT_NB_LAYERS: usize = DEFAULT_NODES_PER_LAYER.len();
/// The default neural network learning rate, for newly created neural networks only
pub const DEFAULT_LEARNING_RATE: NeuralNetworkFloat = 0.000_1;
/// The neural network learning rates evolve with these coefficients after each training session
pub const LEARNING_RATE_EVOLUTIONS: &[NeuralNetworkFloat] = &[0.9, 1., 1.1];
/// Number of learning rates. It is also the number of populations
pub const NB_LEARNING_RATES: usize = LEARNING_RATE_EVOLUTIONS.len();
/// Number of learning iterations after each played game
pub const NB_LEARNINGS_PER_RESULT: usize = 1;
/// Number of neural networks per population
pub const NB_NEURAL_NETWORKS: usize = 8;
/// One neural network is selected every `TOURNAMENT_STEP_SIZE` neural networks to play in the tournament
pub const TOURNAMENT_STEP_SIZE: usize = 1;

pub type NeuralNetwork = cfg_select! {
    base => neural_network::types::neural_network_base::NeuralNetworkBase,
    openblas => neural_network::types::neural_network_openblas::NeuralNetworkOpenblas,
    cublas => neural_network::types::neural_network_cublas::NeuralNetworkCuda,
};

pub type NeuralNetworkFloat = cfg_select! {
    f64_precision => f64,
    not(f64_precision) => f32,
};

pub fn get_mode() -> Mode {
    match std::env::args().nth(1).unwrap().as_str() {
        "play" => Play,
        "train" => Train,
        "tournament" => Tournament,
        mode => panic!("{}", mode),
    }
}
