#[cfg(feature = "openblas")]
extern crate blas_src;

mod players {
    pub mod alpha_beta {
        pub mod alpha_beta;
        pub mod get_move;
        mod score;
    }
    pub mod human {
        pub mod get_move;
    }
    pub mod neural_network {
        pub mod get_move;
        pub mod neural_networks_types {
            pub mod matrix;
            pub mod neural_network;
            #[cfg(feature = "base")]
            pub mod neural_network_base;
            #[cfg(feature = "cublas")]
            pub mod neural_network_cublas;
            #[cfg(feature = "openblas")]
            pub mod neural_network_openblas;
            #[cfg(feature = "cublas")]
            pub mod cuda {
                pub mod cuda;
                pub mod handle_pointer;
                pub mod matrix_pointer;
            }
        }
        pub mod training;
    }
}
mod checkers {
    pub mod bitboard;
    pub mod board;
    pub mod piece;
    pub mod piece_type;
    pub mod player;
    pub mod win_status;
}
mod utils;

#[cfg(all(feature = "base", not(feature = "openblas")))]
pub type NeuralNetwork =
    players::neural_network::neural_networks_types::neural_network_base::NeuralNetworkBase;
#[cfg(feature = "openblas")]
pub type NeuralNetwork =
    players::neural_network::neural_networks_types::neural_network_openblas::NeuralNetworkOpenblas;
#[cfg(feature = "cublas")]
pub type NeuralNetwork =
    players::neural_network::neural_networks_types::neural_network_cublas::NeuralNetworkCuda;

use crate::players::neural_network::training::train_loop;
use std::time::Duration;

// TODO: Solve simple positions (eg: <= 3-4 pieces per player)?
//  Implement Monte Carlo tree search?

const PLAY_AS_WHITE: bool = true;
const PLAY_AS_BLACK: bool = !PLAY_AS_WHITE;
const AI_TIME_PER_MOVE: Duration = Duration::from_secs(1);
const MAX_DEPTH_TRAINING: i8 = 2 * 1;
const MAX_THREADS_DEPTH: i8 = 2; // recommended: 1 or 2 (branch-size usually between 5 and 10)
const BEST_MOVE_FIRST_MIN_DEPTH: i8 = 2 * 4; // use "best move first" strategy if depth >= BEST_MOVE_FIRST_MIN_DEPTH
const BEST_MOVE_FIRST_SKIP_SIZE: i8 = 2 * 3; // choose the best move by exploring at depth := depth - BEST_MOVE_FIRST_SKIP_SIZE

//const NODES_PER_LAYER: [usize; 6] = [130, 512, 1024, 512, 256, 1];
const NODES_PER_LAYER: [usize; 8] = [130, 512, 1024, 1024, 512, 256, 128, 1];
const NB_LAYERS: usize = NODES_PER_LAYER.len();
const LEARNING_RATE_EVOLUTIONS: &[f64] = &[0.5, 0.623, 0.794, 1., 1.260, 1.587, 2.]; // Rounded logscale of 0.5..2.
const NB_LEARNING_RATES: usize = LEARNING_RATE_EVOLUTIONS.len();
const NB_LEARNINGS_PER_RESULT: usize = 10;

fn main() {
    train_loop("neural_networks");
}
