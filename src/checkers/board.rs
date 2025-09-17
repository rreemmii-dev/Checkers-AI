// Rules set: English draughts (https://en.wikipedia.org/wiki/English_draughts)

use crate::checkers::bitboard::BitBoard;
use crate::checkers::piece::Piece;
use crate::checkers::piece_type::PieceType::{King, Man};
use crate::checkers::player::Player;
use crate::checkers::player::Player::{Black, White};
use crate::checkers::win_status::WinStatus;
use crate::checkers::win_status::WinStatus::{Continue, Draw, Win};
use std::collections::HashMap;
use std::process::exit;
use std::vec;

pub type Move = Vec<(i8, i8)>;
pub type BoardHash = (bool, u32, u32, u32);

#[derive(Clone)]
pub struct Board {
    white_bitboard: u32,
    black_bitboard: u32,
    king_bitboard: u32,
    current_player: Player,
    board_count: HashMap<BoardHash, i8>,
    moves_without_capture: i8,
}

pub const BOARD_SIZE: i8 = 8;
pub const NB_PLAYERS_LINES: i8 = 3;
pub const MAX_BOARD_COUNT: i8 = 3;
pub const MAX_MOVES_WITHOUT_CAPTURE: i8 = 2 * 40;

const DIRECTION_KING: &[(i8, i8)] = &[(1, 1), (-1, 1), (1, -1), (-1, -1)];
const DIRECTION_MAN_WHITE: &[(i8, i8)] = &[(1, 1), (-1, 1)];
const DIRECTION_MAN_BLACK: &[(i8, i8)] = &[(1, -1), (-1, -1)];

impl Board {
    pub fn new() -> Board {
        let mut b = Board {
            white_bitboard: 0,
            black_bitboard: 0,
            king_bitboard: 0,
            current_player: White,
            board_count: HashMap::new(),
            moves_without_capture: 0,
        };
        for y in 0..NB_PLAYERS_LINES {
            for x in 0..BOARD_SIZE {
                if is_playable(x, y) {
                    b.set(x, y, Some(Piece::from(White, Man)));
                }
            }
        }
        for y in (BOARD_SIZE - NB_PLAYERS_LINES)..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                if is_playable(x, y) {
                    b.set(x, y, Some(Piece::from(Black, Man)));
                }
            }
        }
        b.incr_board_count();
        b
    }

    pub fn get_white_bitboard(&self) -> u32 {
        self.white_bitboard
    }

    pub fn get_black_bitboard(&self) -> u32 {
        self.black_bitboard
    }

    pub fn get_king_bitboard(&self) -> u32 {
        self.king_bitboard
    }

    fn get_mut_white_bitboard(&mut self) -> &mut u32 {
        &mut self.white_bitboard
    }

    fn get_mut_black_bitboard(&mut self) -> &mut u32 {
        &mut self.black_bitboard
    }

    fn get_mut_king_bitboard(&mut self) -> &mut u32 {
        &mut self.king_bitboard
    }

    pub fn get_player_bitboard(&self, player: Player) -> u32 {
        if player.is_white() {
            self.get_white_bitboard()
        } else {
            self.get_black_bitboard()
        }
    }

    pub fn get_any_bitboard(&self) -> u32 {
        self.get_white_bitboard() | self.get_black_bitboard()
    }

    pub fn get_bitboard(&self, piece: Piece) -> u32 {
        let player = piece.get_player();
        let piece_type = piece.get_piece_type();
        let player_bitboard = self.get_player_bitboard(player);
        let piece_type_bitboard = if piece_type.is_king() {
            self.get_king_bitboard()
        } else {
            !self.get_king_bitboard()
        };
        player_bitboard & piece_type_bitboard
    }

    pub fn hash(&self) -> BoardHash {
        (
            self.get_player_is_white(),
            self.get_white_bitboard(),
            self.get_black_bitboard(),
            self.get_king_bitboard(),
        )
    }

    pub fn get(&self, x: i8, y: i8) -> Option<Piece> {
        let is_white = self.get_white_bitboard().is_some(x, y);
        let is_black = self.get_black_bitboard().is_some(x, y);
        let is_king = self.get_king_bitboard().is_some(x, y);
        let is_any = is_white | is_black;

        assert!(!(is_white && is_black));
        assert!(!(is_king && !is_any));

        if is_any {
            Some(Piece::from(
                if is_white { White } else { Black },
                if is_king { King } else { Man },
            ))
        } else {
            None
        }
    }

    fn set(&mut self, x: i8, y: i8, piece_opt: Option<Piece>) {
        let is_white = if let Some(piece) = piece_opt {
            piece.is_white()
        } else {
            false
        };
        let is_black = if let Some(piece) = piece_opt {
            piece.is_black()
        } else {
            false
        };
        let is_king = if let Some(piece) = piece_opt {
            piece.is_king()
        } else {
            false
        };
        self.get_mut_white_bitboard().set(x, y, is_white);
        self.get_mut_black_bitboard().set(x, y, is_black);
        self.get_mut_king_bitboard().set(x, y, is_king);
    }

    pub fn get_player(&self) -> Player {
        self.current_player
    }

    pub fn get_player_is_white(&self) -> bool {
        self.get_player().is_white()
    }

    fn switch_player(&mut self) {
        self.current_player = self.get_player().other();
    }

    pub fn get_board_count(&self) -> i8 {
        match self.board_count.get(&self.hash()) {
            Some(&n) => n,
            None => 0,
        }
    }

    fn incr_board_count(&mut self) {
        self.board_count
            .entry(self.hash())
            .and_modify(|e| *e += 1)
            .or_insert(1);
    }

    fn reset_board_count(&mut self) {
        self.board_count.clear();
    }

    pub fn get_moves_without_capture(&self) -> i8 {
        self.moves_without_capture
    }

    fn incr_moves_without_capture(&mut self) {
        self.moves_without_capture += 1;
    }

    fn reset_moves_without_capture(&mut self) {
        self.moves_without_capture = 0;
    }

    fn is_draw(&self) -> bool {
        self.get_board_count() == MAX_BOARD_COUNT
            || self.get_moves_without_capture() == MAX_MOVES_WITHOUT_CAPTURE
    }

    pub fn get_win_status(&self) -> WinStatus {
        if self.is_draw() {
            Draw
        } else if !(self.can_move() || self.can_jump()) {
            Win(self.get_player().other())
        } else {
            Continue
        }
    }

    pub fn is_end_game(&self) -> bool {
        self.get_win_status().is_end_game()
    }

    pub fn get_piece_counter(&self, piece: Piece) -> u32 {
        let bitboard = self.get_bitboard(piece);
        bitboard.count_ones()
    }

    fn can_move(&self) -> bool {
        let man_directions = if self.get_player_is_white() {
            DIRECTION_MAN_WHITE
        } else {
            DIRECTION_MAN_BLACK
        };
        let current_player = self.get_player();
        let player_bitboard = self.get_player_bitboard(current_player);
        let any_bitboard = self.get_any_bitboard();
        for direction in man_directions {
            if (player_bitboard.move_direction(direction) & !any_bitboard) != 0 {
                return true;
            }
        }
        let player_king_bitboard = player_bitboard & self.get_king_bitboard();
        for direction in DIRECTION_KING {
            if (player_king_bitboard.move_direction(direction) & !any_bitboard) != 0 {
                return true;
            }
        }
        false
    }

    fn can_jump(&self) -> bool {
        let man_directions = if self.get_player_is_white() {
            DIRECTION_MAN_WHITE
        } else {
            DIRECTION_MAN_BLACK
        };
        let current_player = self.get_player();
        let player_bitboard = self.get_player_bitboard(current_player);
        let opponent_bitboard = self.get_player_bitboard(current_player.other());
        let any_bitboard = self.get_any_bitboard();
        for direction in man_directions {
            if (player_bitboard
                .move_direction(direction)
                .move_direction(direction)
                & opponent_bitboard.move_direction(direction)
                & !any_bitboard)
                != 0
            {
                return true;
            }
        }
        let player_king_bitboard = player_bitboard & self.get_king_bitboard();
        for direction in DIRECTION_KING {
            if (player_king_bitboard
                .move_direction(direction)
                .move_direction(direction)
                & opponent_bitboard.move_direction(direction)
                & !any_bitboard)
                != 0
            {
                return true;
            }
        }
        false
    }

    pub fn possible_moves(&self) -> Vec<Move> {
        if self.is_draw() {
            return Vec::new();
        }

        fn add_moving(board: &Board, moves: &mut Vec<Move>, directions: &[(i8, i8)], x: i8, y: i8) {
            for &(dx, dy) in directions {
                let (x2, y2) = (x + dx, y + dy);
                if is_playable(x2, y2) && board.get_any_bitboard().is_none(x2, y2) {
                    moves.push(vec![(x, y), (x2, y2)]);
                }
            }
        }

        fn add_jumping(
            board: &mut Board,
            moves: &mut Vec<Move>,
            directions: &[(i8, i8)],
            x: i8,
            y: i8,
        ) -> Vec<usize> {
            let mut moves_indexes = Vec::new();
            for &(dx, dy) in directions {
                let (x2, y2) = (x + dx, y + dy);
                let (x3, y3) = (x2 + dx, y2 + dy);
                if is_playable(x3, y3)
                    && board.get_any_bitboard().is_none(x3, y3)
                    && board
                        .get_player_bitboard(board.get_player().other())
                        .is_some(x2, y2)
                {
                    let jumping_piece = board.get(x, y);
                    let taken_piece = board.get(x2, y2);
                    board.set(x, y, None);
                    board.set(x2, y2, None);
                    board.set(x3, y3, jumping_piece);
                    let mut further_moves_indexes = add_jumping(board, moves, directions, x3, y3);
                    if further_moves_indexes.is_empty() {
                        moves.push(vec![(x3, y3)]);
                        further_moves_indexes.push(moves.len() - 1);
                    }
                    for &move_index in &further_moves_indexes {
                        moves[move_index].insert(0, (x, y));
                    }

                    board.set(x, y, jumping_piece);
                    board.set(x2, y2, taken_piece);
                    board.set(x3, y3, None);

                    moves_indexes.extend(further_moves_indexes);
                }
            }
            moves_indexes
        }

        let mut board = self.clone();
        let mut moves = Vec::new();
        let can_jump = board.can_jump();

        for x in 0..BOARD_SIZE {
            for y in 0..BOARD_SIZE {
                if !is_playable(x, y) {
                    continue;
                }
                if board.get_player_bitboard(board.get_player()).is_none(x, y) {
                    continue;
                }
                let piece_opt = self.get(x, y);
                let directions = allowed_directions(piece_opt.unwrap());
                if can_jump {
                    add_jumping(&mut board, &mut moves, directions, x, y);
                } else {
                    add_moving(&mut board, &mut moves, directions, x, y);
                }
            }
        }

        moves.sort();
        moves
    }

    pub fn play(&mut self, moves: &Move) {
        let win_status = self.get_win_status();
        if win_status.is_end_game() {
            println!("Game Over!");
        }
        if let Win(player) = win_status {
            println!("> {player:?} won!");
        } else if win_status == Draw {
            println!("> Draw");
        }
        if win_status.is_end_game() {
            exit(0);
        }

        let possible_moves = self.possible_moves();
        assert!(
            possible_moves.contains(moves),
            "Unauthorized move ({moves:?}). List: {possible_moves:?}"
        );

        let (mut x, mut y) = moves[0];
        let (x2, y2) = moves[1];
        if i8::abs(y2 - y) == 1 {
            // No jump
            self.set(x2, y2, self.get(x, y));
            self.set(x, y, None);
            (x, y) = (x2, y2);
            if self.get(x2, y2).unwrap().is_man() {
                self.reset_board_count();
            }
            self.incr_moves_without_capture();
        } else {
            // Jump
            for &(x2, y2) in &moves[1..] {
                let (dx, dy) = ((x2 - x) / 2, (y2 - y) / 2);
                let (x_mid, y_mid) = (x + dx, y + dy);
                self.set(x_mid, y_mid, None);
                self.set(x2, y2, self.get(x, y));
                self.set(x, y, None);
                (x, y) = (x2, y2);
            }
            self.reset_board_count();
            self.reset_moves_without_capture();
        }
        if y == BOARD_SIZE - 1 && self.get(x, y).unwrap() == (Piece::from(White, Man)) {
            self.set(x, y, Some(Piece::from(White, King)));
            self.reset_board_count();
            self.reset_moves_without_capture();
        } else if y == 0 && self.get(x, y).unwrap() == (Piece::from(Black, Man)) {
            self.set(x, y, Some(Piece::from(Black, King)));
            self.reset_board_count();
            self.reset_moves_without_capture();
        }
        self.switch_player();

        self.incr_board_count();
    }

    pub fn display(&self) {
        println!("\n{:?} is playing", self.get_player());
        println!(
            "Moves without capture nor promotion: {}/{}",
            self.get_moves_without_capture(),
            MAX_MOVES_WITHOUT_CAPTURE
        );
        println!(
            "Board count: {}/{}",
            self.get_board_count(),
            MAX_BOARD_COUNT
        );
        println!(
            "White: {} men, {} kings",
            self.get_piece_counter(Piece::from(White, Man)),
            self.get_piece_counter(Piece::from(White, King)),
        );
        println!(
            "Black: {} men, {} kings",
            self.get_piece_counter(Piece::from(Black, Man)),
            self.get_piece_counter(Piece::from(Black, King)),
        );
        println!();
        print!("   ");
        for x in 0..BOARD_SIZE {
            print!(" {} ", char_of_x(x));
        }
        println!();
        for y in (0..BOARD_SIZE).rev() {
            print!(" {} ", char_of_y(y));
            for x in 0..BOARD_SIZE {
                if is_playable(x, y) {
                    match self.get(x, y) {
                        Some(piece) => print!(" {} ", piece.emoji()),
                        None => print!(" • "),
                    }
                } else {
                    print!("   ");
                }
            }
            print!(" {} ", char_of_y(y));
            println!();
        }
        print!("   ");
        for x in 0..BOARD_SIZE {
            print!(" {} ", char_of_x(x));
        }
        println!("\n");
    }
}

pub fn is_playable(x: i8, y: i8) -> bool {
    (0..BOARD_SIZE).contains(&x) && (0..BOARD_SIZE).contains(&y) && (x + y) % 2 == 0
}

fn allowed_directions(piece: Piece) -> &'static [(i8, i8)] {
    if piece.is_king() {
        DIRECTION_KING
    } else if piece.is_white() {
        DIRECTION_MAN_WHITE
    } else {
        DIRECTION_MAN_BLACK
    }
}

pub fn char_of_x(x: i8) -> char {
    (x as u8 + 'A' as u8) as char
}

pub fn char_of_y(y: i8) -> char {
    (y as u8 + '1' as u8) as char
}
