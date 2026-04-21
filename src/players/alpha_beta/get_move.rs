use crate::checkers::board::{Board, Move};
use crate::players::alpha_beta::score::{BLACK_SIGN, WHITE_SIGN, naive_score};
use crate::players::utils::alpha_beta::alpha_beta_moves_list;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::thread::sleep;
use std::time::Duration;

pub fn get_alpha_beta_move_simple_heuristic_time_limit(
    board: &Board,
    duration: Duration,
    threaded: bool,
) -> Move {
    get_alpha_beta_move_time_limit(board, Arc::new(simple_heuristic), duration, threaded)
}

pub fn get_alpha_beta_move_depth_limit(
    board: &Board,
    // TODO: Use a pub type `Arc<...>` everywhere when being part of stable Rust
    #[cfg(nn_is_sync)] heuristic: Arc<impl Fn(&Board) -> i64 + Send + Sync + 'static>,
    #[cfg(not(nn_is_sync))] heuristic: Arc<impl Fn(&Board) -> i64 + Send + Clone + 'static>,
    max_depth: i8,
    threaded: bool,
) -> Move {
    let cancel_search = Arc::new(AtomicBool::new(false));
    let best_moves =
        alpha_beta_moves_list(board, heuristic, max_depth, threaded, cancel_search).unwrap();
    let i = rand::random_range(0..best_moves.len());
    best_moves[i].clone()
}

pub fn get_alpha_beta_move_time_limit(
    board: &Board,
    #[cfg(nn_is_sync)] heuristic: Arc<impl Fn(&Board) -> i64 + Send + Sync + 'static>,
    #[cfg(not(nn_is_sync))] heuristic: Arc<impl Fn(&Board) -> i64 + Send + Clone + 'static>,
    duration: Duration,
    threaded: bool,
) -> Move {
    let mut best_moves = Vec::new();
    let mut depth = 2 * 1;
    let cancel_search = Arc::new(AtomicBool::new(false));
    {
        let cancel_search = cancel_search.clone();
        thread::spawn(move || {
            sleep(duration);
            cancel_search.store(true, Ordering::Release);
        });
    }
    while !cancel_search.load(Ordering::Acquire) {
        let heuristic = heuristic.clone();
        let cancel_search = cancel_search.clone();
        let new_best_moves_opt =
            alpha_beta_moves_list(board, heuristic, depth, threaded, cancel_search);
        if let Some(new_best_moves) = new_best_moves_opt {
            best_moves = new_best_moves;
            depth += 2;
            if depth >= 2 * 50 {
                // Probably end of game, or only 1 move allowed
                break;
            }
        }
    }
    let i = rand::random_range(0..best_moves.len());
    best_moves[i].clone()
}

pub fn simple_heuristic(board: &Board) -> i64 {
    // Current player POV
    (if board.get_player_is_white() {
        WHITE_SIGN
    } else {
        BLACK_SIGN
    }) * naive_score(board)
}
