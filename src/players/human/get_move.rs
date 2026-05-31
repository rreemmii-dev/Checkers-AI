use crate::checkers::board::{Board, Move, char_of_x, char_of_y};
use crate::players::alpha_beta::get_move::{get_alpha_beta_move_depth_limit, simple_heuristic};
use std::fmt::Write;
use std::io::stdin;
use std::sync::Arc;

pub fn get_human_move(board: &Board) -> Move {
    let possible_moves = board.possible_moves();
    if possible_moves.is_empty() {
        return Vec::new();
    }
    println!("Possible moves:");
    print_moves_list(&possible_moves);
    println!("Which move do you want to play?");
    let mut good_move_index = 0;
    let good_move = get_alpha_beta_move_depth_limit(board, Arc::new(simple_heuristic), 2, false);
    for i in 0..possible_moves.len() {
        if possible_moves[i] == good_move {
            good_move_index = i;
            break;
        }
    }
    println!(
        "For example, press {} then enter to play the move `{}: {}`",
        good_move_index,
        good_move_index,
        string_of_move(&possible_moves[good_move_index])
    );
    let mut input = String::new();
    if stdin().read_line(&mut input).is_err() {
        return get_human_move(board);
    }
    if let Ok(index) = input.trim().parse::<usize>() {
        if index >= possible_moves.len() {
            return get_human_move(board);
        }
        let vec = possible_moves[index].clone();
        println!("Play {}", string_of_move(&vec));
        vec
    } else {
        get_human_move(board)
    }
}

fn print_moves_list(moves: &[Move]) {
    for (i, m) in moves.iter().enumerate() {
        println!("{}: {}", i, string_of_move(m));
    }
}

fn string_of_move(m: &Move) -> String {
    let mut str = String::new();
    for &(x, y) in m {
        write!(str, "{}{} -> ", char_of_x(x), char_of_y(y)).unwrap();
    }
    for _ in 0.." -> ".len() {
        str.pop();
    }
    str
}
