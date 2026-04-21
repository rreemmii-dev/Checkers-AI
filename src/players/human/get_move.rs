use crate::checkers::board::{Board, Move, char_of_x, char_of_y};
use std::fmt::Write;
use std::io::stdin;

pub fn get_human_move(board: &Board) -> Move {
    let possible_moves = board.possible_moves();
    if possible_moves.is_empty() {
        return Vec::new();
    }
    print_moves_list(&possible_moves);
    println!("Which move?");
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
