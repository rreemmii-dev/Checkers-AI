use crate::checkers::board::Board;
use crate::players::alpha_beta::alpha_beta::{alpha_beta_list, threaded_moves_list};
use crate::players::alpha_beta::score::{BLACK_SIGN, WHITE_SIGN, naive_score};
use crate::{
    AI_TIME_PER_MOVE, BEST_MOVE_FIRST_MIN_DEPTH, BEST_MOVE_FIRST_SKIP_SIZE, MAX_THREADS_DEPTH,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::thread::sleep;

fn heuristic_score(board: &Board) -> i64 {
    // Current player POV
    (if board.get_player_is_white() {
        WHITE_SIGN
    } else {
        BLACK_SIGN
    }) * naive_score(board)
}

pub fn get_alpha_beta_move(board: &Board, threaded: bool) -> Vec<(i8, i8)> {
    if board.is_end_game() {
        return Vec::new();
    }

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
        let new_best_moves_opt = if threaded {
            threaded_moves_list(
                board.clone(),
                Arc::new(heuristic_score),
                BEST_MOVE_FIRST_MIN_DEPTH,
                BEST_MOVE_FIRST_SKIP_SIZE,
                depth,
                MAX_THREADS_DEPTH,
                cancel_search.clone(),
            )
        } else {
            alpha_beta_list(
                &board,
                &heuristic_score,
                BEST_MOVE_FIRST_MIN_DEPTH,
                BEST_MOVE_FIRST_SKIP_SIZE,
                depth,
                &cancel_search,
            )
        };
        if let Some(new_best_moves) = new_best_moves_opt {
            best_moves = new_best_moves;
            depth += 2;
            if depth >= 2 * 50 {
                // Probably end of game, or only 1 move allowed
                break;
            }
        }
    }
    let x = rand::random_range(0..best_moves.len());
    best_moves[x].clone()
}
