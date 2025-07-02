use crate::checkers::Board;
use crate::score::{BLACK_SIGN, NEG_INFINITY, POS_INFINITY, WHITE_SIGN, naive_score};
use std::sync::Arc;
use std::thread;

fn board_score(board: &Board) -> i64 {
    (if board.get_player_is_white() {
        WHITE_SIGN
    } else {
        BLACK_SIGN
    }) * naive_score(board)
}

fn alpha_beta_score(board: &Board, alpha: i64, beta: i64, depth: i8) -> i64 {
    if depth == 0 || board.is_end_game() {
        return board_score(board);
    }
    let mut alpha = alpha;
    for m in board.possible_moves() {
        let mut cloned_board = board.clone();
        cloned_board.play(&m);
        let res = -alpha_beta_score(&cloned_board, -beta, -alpha, depth - 1);
        if res >= beta {
            return beta;
        } else if res > alpha {
            alpha = res;
        }
    }
    alpha
}

fn alpha_beta_list(board: &Board, depth: i8) -> (i64, Vec<Vec<(i8, i8)>>) {
    if depth == 0 || board.is_end_game() {
        return (board_score(board), Vec::new());
    }
    let mut best_score = NEG_INFINITY;
    let mut best_moves = Vec::new();
    for m in board.possible_moves() {
        let mut cloned_board = board.clone();
        cloned_board.play(&m);
        let res = -alpha_beta_score(&cloned_board, -POS_INFINITY, -best_score, depth - 1);
        if res > best_score {
            best_score = res;
            best_moves = vec![m.clone()];
        } else if res == best_score {
            best_moves.push(m.clone());
        }
    }
    (best_score, best_moves)
}

fn threaded_score(board: Board, depth: i8, threads_depth: i8) -> i64 {
    if threads_depth == 0 {
        return alpha_beta_score(&board, NEG_INFINITY, POS_INFINITY, depth);
    }
    if depth == 0 || board.is_end_game() {
        return board_score(&board);
    }
    let mut handle = Vec::new();
    let board = Arc::new(board);
    for m in board.possible_moves() {
        let board = board.clone();
        handle.push(thread::spawn(move || {
            let mut cloned_board = (*board).clone();
            cloned_board.play(&m);
            -threaded_score(cloned_board, depth - 1, threads_depth - 1)
        }));
    }
    let mut best_score = NEG_INFINITY;
    for h in handle {
        let score = h.join().unwrap();
        if score > best_score {
            best_score = score;
        }
    }
    best_score
}

pub fn threaded_moves_list(
    board: Board,
    depth: i8,
    threads_depth: i8,
) -> (i64, Vec<Vec<(i8, i8)>>) {
    if threads_depth == 0 {
        return alpha_beta_list(&board, depth);
    }
    if depth == 0 || board.is_end_game() {
        return (board_score(&board), Vec::new());
    }
    let mut handle = Vec::new();
    let board = Arc::new(board);
    for m in board.possible_moves() {
        let board = board.clone();
        handle.push(thread::spawn(move || {
            let mut cloned_board = (*board).clone();
            cloned_board.play(&m);
            let res = -threaded_score(cloned_board, depth - 1, threads_depth - 1);
            (res, m)
        }));
    }
    let mut best_score = NEG_INFINITY;
    let mut best_moves = Vec::new();
    for h in handle {
        let (score, m) = h.join().unwrap();
        if score > best_score {
            best_score = score;
            best_moves = vec![m];
        } else if score == best_score {
            best_moves.push(m);
        }
    }
    (best_score, best_moves)
}
