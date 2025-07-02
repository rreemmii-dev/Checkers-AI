mod alpha_beta;
mod checkers;
mod score;

use crate::alpha_beta::threaded_moves_list;
use crate::checkers::{Board, char_of_x, char_of_y};
use std::fmt::Write;
use std::time::Instant;

const PLAY_AS_WHITE: bool = true;
const MAX_DEPTH: i8 = 2 * 5; // recommended: 2 * 5 in dev, 2 * 6 in release
const MAX_THREADS_DEPTH: i8 = 2; // recommended: 2 (branch-size between 5 and 10, so creates between 25 and 100 threads)

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
    // let (_score, best_moves) = alpha_beta_list(&board, MAX_DEPTH);
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
            // get_ai_move(&b)
        } else {
            get_ai_move(&b)
        };
        b.play(&vec);

        b.display();
        let vec = if PLAY_AS_WHITE {
            get_ai_move(&b)
        } else {
            get_human_move(&b)
            // get_ai_move(&b)
        };
        b.play(&vec);
    }
}
