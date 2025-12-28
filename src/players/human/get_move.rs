use crate::checkers::board::Board;
use crate::utils::{print_moves_list, string_of_move};

pub fn get_human_move(board: &Board) -> Vec<(i8, i8)> {
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
