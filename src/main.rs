#[cfg(openblas)]
extern crate blas_src;

mod checkers {
    pub mod bitboard;
    pub mod board;
    pub mod piece;
    pub mod piece_type;
    pub mod player;
    pub mod win_status;
}
pub mod neural_network {
    pub mod training {
        pub mod graphs;
        pub mod heuristic_comparison;
        pub mod tournament;
        pub mod train;
    }
    pub mod types {
        #[cfg(cublas)]
        pub mod cuda {
            pub mod context_pointer;
            pub mod cuda;
            pub mod matrix_pointer;
        }
        pub mod matrix;
        pub mod neural_network_base;
        #[cfg(cublas)]
        pub mod neural_network_cublas;
        #[cfg(openblas)]
        pub mod neural_network_openblas;
    }
    pub mod neural_network;
    pub mod storage;
}
mod players {
    pub mod alpha_beta {
        pub mod get_move;
        pub mod score;
    }
    pub mod human {
        pub mod get_move;
    }
    pub mod neural_network {
        pub mod get_move;
    }
    pub mod utils {
        pub mod alpha_beta;
    }
}
mod consts;

use std::thread::sleep;
use std::time::Duration;
use crate::checkers::board::Board;
use crate::checkers::player::Player::{Black, White};
use crate::checkers::win_status::WinStatus::{Continue, Draw, Win};
use crate::consts::Mode::{Play, Tournament, Train};
use crate::consts::{AI_TIME_LIMIT_STRATEGY, get_mode};
use crate::neural_network::storage::load_neural_network;
use crate::neural_network::training::tournament::run_tournament;
use crate::neural_network::training::train::train_loop;
use crate::players::human::get_move::get_human_move;
use crate::players::neural_network::get_move::get_neural_network_move;

fn main() {
    match get_mode() {
        Play => play(),
        Train => train_loop("neural_networks"),
        Tournament => run_tournament("neural_networks"),
    }
}

fn play() {
    let neural_network = load_neural_network("neural_network.txt");
    loop {
        let mut board = Board::new();
        while !board.is_end_game() {
            println!("{}", board);
            let m = if board.get_player_is_white() {
                get_human_move(&board)
            } else {
                get_neural_network_move(&board, &neural_network, AI_TIME_LIMIT_STRATEGY, true)
            };
            board.play(&m);
        }
        println!("{}", board);
        match board.get_win_status() {
            Win(White) => println!("> You won!"),
            Win(Black) => println!("> You lost (the game)!"),
            Draw => println!("> Draw!"),
            Continue => panic!("Continue"),
        }
        sleep(Duration::from_secs(5));
    }
}
