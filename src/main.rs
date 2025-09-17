mod ai_algorithms {
    pub mod alpha_beta;
}
mod checkers {
    pub mod bitboard;
    pub mod board;
    pub mod piece;
    pub mod piece_type;
    pub mod player;
    pub mod win_status;
}
mod score;

use crate::ai_algorithms::alpha_beta::threaded_moves_list;
use crate::checkers::board::{Board, char_of_x, char_of_y};
use crate::score::{BLACK_SIGN, WHITE_SIGN, naive_score};
use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::thread::sleep;
use std::time::{Duration, Instant};

const PLAY_AS_WHITE: bool = true;
const PLAY_AS_BLACK: bool = !PLAY_AS_WHITE;
const AI_TIME_PER_MOVE: Duration = Duration::from_secs(1);
const MAX_THREADS_DEPTH: i8 = 1; // recommended: 1 or 2 (branch-size usually between 5 and 10)
const BEST_MOVE_FIRST_MIN_DEPTH: i8 = 2 * 4; // use "best move first" strategy if depth >= BEST_MOVE_FIRST_MIN_DEPTH
const BEST_MOVE_FIRST_SKIP_SIZE: i8 = 2 * 3; // choose the best move by exploring at depth := depth - BEST_MOVE_FIRST_SKIP_SIZE

fn string_of_move(m: &Vec<(i8, i8)>) -> String {
    let mut str = String::new();
    for &(x, y) in m {
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
    if let Err(_) = std::io::stdin().read_line(&mut input) {
        return get_human_move(&board);
    }
    if let Ok(index) = input.trim().parse::<usize>() {
        if index >= possible_moves.len() {
            return get_human_move(&board);
        }
        let vec = possible_moves[index].clone();
        println!("Play {}", string_of_move(&vec));
        vec
    } else {
        get_human_move(board)
    }
}

fn heuristic_score(board: &Board) -> i64 {
    // Current player POV
    (if board.get_player_is_white() {
        WHITE_SIGN
    } else {
        BLACK_SIGN
    }) * naive_score(board)
}

fn get_ai_move(board: &Board) -> Vec<(i8, i8)> {
    print_moves_list(&board.possible_moves());
    if board.is_end_game() {
        return Vec::new();
    }
    let time = Instant::now();

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
        if let Some((_, new_best_moves)) = threaded_moves_list(
            board.clone(),
            Arc::new(heuristic_score),
            BEST_MOVE_FIRST_MIN_DEPTH,
            BEST_MOVE_FIRST_SKIP_SIZE,
            depth,
            MAX_THREADS_DEPTH,
            cancel_search.clone(),
        ) {
            best_moves = new_best_moves;
            depth += 2;
        }
    }
    println!("AI depth: 2 * {}", depth / 2);
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
        let vec = if b.get_player().is_white() && PLAY_AS_WHITE
            || b.get_player().is_black() && PLAY_AS_BLACK
        {
            get_human_move(&b)
        } else {
            get_ai_move(&b)
        };
        b.play(&vec);
        if b.is_end_game() {
            b.display();
            break;
        }
    }
}
