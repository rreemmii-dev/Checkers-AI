mod checkers;

use crate::checkers::{char_of_x, char_of_y, Board, Piece, BOARD_SIZE, NB_PLAYERS_LINES, MAX_BOARD_COUNT, MAX_MOVES_WITHOUT_CAPTURE};

use std::fmt::Write;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const MAN_SCORE: i64 = 1;
const KING_SCORE: i64 = 2;
const WHITE_SIGN: i64 = 1;
const BLACK_SIGN: i64 = -1;
const MAX_BOARD_COUNT_SCORE_COEF: i64 = 100;
const MAX_MOVES_WITHOUT_CAPTURE_SCORE_COEF: i64 = 100;
const MAX_SCORE_COEF: i64 = MAX_BOARD_COUNT_SCORE_COEF * MAX_MOVES_WITHOUT_CAPTURE_SCORE_COEF;
const MAX_SCORE_WITHOUT_COEF: i64 = (NB_PLAYERS_LINES * BOARD_SIZE / 2) as i64 * KING_SCORE;
const POS_INFINITY: i64 = MAX_SCORE_WITHOUT_COEF * MAX_SCORE_COEF + 1;
const NEG_INFINITY: i64 = -POS_INFINITY;
const WHITE_WIN: i64 = WHITE_SIGN * (POS_INFINITY - 1);
const BLACK_WIN: i64 = BLACK_SIGN * (POS_INFINITY - 1);
const DRAW: i64 = 0;
const MAX_DEPTH: i8 = 2 * 6; // recommended: 2 * 5 in dev, 2 * 6 in release
const MAX_THREADS_DEPTH: i8 = 2; // recommended: 2 (branch-size between 5 and 10, so creates between 25 and 100 threads)
const PLAY_AS_WHITE: bool = true;

fn piece_score(piece: Piece) -> i64 {
    let sign = if piece.is_white() { WHITE_SIGN } else { BLACK_SIGN };
    let value = if piece.is_king() { KING_SCORE } else { MAN_SCORE };
    sign * value
}

fn naive_score(board: &Board) -> i64 {
    if board.is_draw() {
        return DRAW;
    }
    if board.possible_moves().is_empty() {
        return if board.get_player_is_white() {
            BLACK_WIN
        } else {
            WHITE_WIN
        };
    }

    fn coef_board_count(n: i8) -> i64 {
        assert_eq!(MAX_BOARD_COUNT, 3);
        if n == 0 || n == 1 {
            MAX_BOARD_COUNT_SCORE_COEF * 1
        } else if n == 2 {
            MAX_BOARD_COUNT_SCORE_COEF * 3 / 4
        } else if n == 3 {
            MAX_BOARD_COUNT_SCORE_COEF * 0
        } else {
            panic!()
        }
    }

    fn coef_moves_without_capture(n: i8) -> i64 {
        assert_eq!(MAX_MOVES_WITHOUT_CAPTURE, 80);
        let n = i64::from(n);
        if n <= 40 {
            MAX_MOVES_WITHOUT_CAPTURE_SCORE_COEF * 1
        } else if n <= 60 {
            MAX_MOVES_WITHOUT_CAPTURE_SCORE_COEF * (100 - (n - 40) * 1) / 100
        } else if n <= 80 {
            MAX_MOVES_WITHOUT_CAPTURE_SCORE_COEF * (100 - 20 - (n - 60) * 4) / 100
        } else {
            panic!()
        }
    }

    let mut score = 0;
    for (piece_hash, count) in board.get_pieces_counter().into_iter().enumerate() {
        score += count * piece_score(Piece::unhash(piece_hash));
    }
    score *= coef_board_count(board.get_board_count());
    score *= coef_moves_without_capture(board.get_moves_without_capture());
    score
}

fn alpha_beta_score(board: &Board, alpha: i64, beta: i64, depth: i8) -> i64 {
    if depth == 0 || board.possible_moves().is_empty() {
        return if board.get_player_is_white() { WHITE_SIGN } else { BLACK_SIGN } * naive_score(board);
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
    if depth == 0 || board.possible_moves().is_empty() {
        let score = if board.get_player_is_white() { WHITE_SIGN } else { BLACK_SIGN } * naive_score(board);
        return (score, Vec::new());
    }
    let mut best_score = NEG_INFINITY;
    let mut best_moves = Vec::new();
    for m in board.possible_moves() {
        let mut cloned_board = board.clone();
        cloned_board.play(&m);
        let res = -alpha_beta_score(&cloned_board, best_score - 1, POS_INFINITY, depth - 1);
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
    if depth == 0 || board.possible_moves().is_empty() {
        let score = if board.get_player_is_white() { WHITE_SIGN } else { BLACK_SIGN } * naive_score(&board);
        return score;
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

fn threaded_moves_list(board: Board, depth: i8, threads_depth: i8) -> (i64, Vec<Vec<(i8, i8)>>) {
    if threads_depth == 0 {
        return alpha_beta_list(&board, depth);
    }
    if depth == 0 || board.possible_moves().is_empty() {
        let score = if board.get_player_is_white() { WHITE_SIGN } else { BLACK_SIGN } * naive_score(&board);
        return (score, Vec::new());
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

fn string_of_move(m: &Vec<(i8, i8)>) -> String {
    let mut str = String::new();
    for &(x, y) in m.iter().rev() {
        write!(str, "{}{} -> ", char_of_x(x), char_of_y(y)).unwrap();
    }
    for _ in 0.." -> ".len() {
        str.pop();
    }
    str
}

fn print_moves_list(moves: &Vec<Vec<(i8, i8)>>) {
    for (i, m) in moves.iter().enumerate() {
        println!("{}: {}", i, string_of_move(m));
    }
}

fn get_human_move(board: &Board) -> Vec<(i8, i8)> {
    let possible_moves = board.possible_moves();
    if possible_moves.is_empty() {
        return Vec::new();
    }
    print_moves_list(&possible_moves);
    println!("Which move?");
    let mut input = String::new();
    match std::io::stdin().read_line(&mut input) {
        Ok(_) => {
            let index = input.trim().parse::<usize>().unwrap();
            if index < possible_moves.len() {
                let vec = possible_moves[index].clone();
                println!("Play {}", string_of_move(&vec));
                vec
            } else {
                get_human_move(board)
            }
        }
        Err(_) => get_human_move(board),
    }
}

fn get_ai_move(board: &Board) -> Vec<(i8, i8)> {
    print_moves_list(&board.possible_moves());
    if board.possible_moves().is_empty() {
        return Vec::new();
    }
    let time = Instant::now();
    let (_score, best_moves) = threaded_moves_list(board.clone(), MAX_DEPTH, MAX_THREADS_DEPTH);
    let dt = time.elapsed();
    println!("AI computation time: {dt:?}");
    let x = rand::random_range(0..best_moves.len());
    println!("Play {}", string_of_move(&best_moves[x]));
    best_moves[x].clone()
}

fn main() {
    let mut b = Board::new();

    loop {
        b.display();
        let vec = if PLAY_AS_WHITE {
            get_human_move(&b)
        } else {
            get_ai_move(&b)
        };
        b.play(&vec);

        b.display();
        let vec = if PLAY_AS_WHITE {
            get_ai_move(&b)
        } else {
            get_human_move(&b)
        };
        b.play(&vec);
    }
}
