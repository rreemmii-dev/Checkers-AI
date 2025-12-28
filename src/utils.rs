use crate::checkers::board::{char_of_x, char_of_y};
use std::fmt::Write;

pub fn string_of_move(m: &Vec<(i8, i8)>) -> String {
    let mut str = String::new();
    for &(x, y) in m {
        write!(str, "{}{} -> ", char_of_x(x), char_of_y(y)).unwrap();
    }
    for _ in 0.." -> ".len() {
        str.pop();
    }
    str
}

pub fn print_moves_list(moves: &Vec<Vec<(i8, i8)>>) {
    for (i, m) in moves.iter().enumerate() {
        println!("{}: {}", i, string_of_move(m));
    }
}
