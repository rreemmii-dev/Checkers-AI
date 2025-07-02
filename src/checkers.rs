// Rules set: English draughts (https://en.wikipedia.org/wiki/English_draughts)

use crate::checkers::PieceType::{King, Man};
use crate::checkers::Player::{Black, White};
use crate::checkers::WinStatus::{Continue, Draw, Win};
use std::cmp::PartialEq;
use std::collections::HashMap;
use std::process::exit;
use std::vec;

pub const BOARD_SIZE: i8 = 8;
pub const NB_PLAYERS_LINES: i8 = 3;
pub const MAX_BOARD_COUNT: i8 = 3;
pub const MAX_MOVES_WITHOUT_CAPTURE: i8 = 2 * 40;

const MAX_PIECE_HASH_VALUE: usize = 3;
const NONE_PIECE_HASH_VALUE: usize = MAX_PIECE_HASH_VALUE + 1;
const MAX_OPT_PIECE_HASH_VALUE: usize = NONE_PIECE_HASH_VALUE;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Player {
    White,
    Black,
}

#[derive(PartialEq)]
pub enum WinStatus {
    Win(Player),
    Draw,
    Continue,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum PieceType {
    Man,
    King,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Piece {
    player: Player,
    piece_type: PieceType,
}

pub type Move = Vec<(i8, i8)>;
pub type BoardHash = (bool, u64, u64, u64);
type BoardMatrix = [[Option<Piece>; BOARD_SIZE as usize]; BOARD_SIZE as usize];

#[derive(Clone)]
pub struct Board {
    board: BoardMatrix,
    current_player: Player,
    board_count: HashMap<BoardHash, i8>,
    moves_without_capture: i8,
}

impl Player {
    fn other(self) -> Player {
        match self {
            White => Black,
            Black => White,
        }
    }

    pub fn is_white(self) -> bool {
        self == White
    }

    pub fn is_black(self) -> bool {
        self == Black
    }
}

impl WinStatus {
    pub fn is_draw(self) -> bool {
        self == Draw
    }

    pub fn is_win(self) -> bool {
        matches!(self, Win(_))
    }

    pub fn is_end_game(self) -> bool {
        self != Continue
    }

    pub fn get_win(self) -> Option<Player> {
        if let Win(player) = self {
            Some(player)
        } else {
            None
        }
    }
}

impl Piece {
    pub fn hash(self) -> usize {
        assert_eq!(MAX_PIECE_HASH_VALUE, 3);
        match self {
            Piece {
                player: White,
                piece_type: Man,
            } => 0,
            Piece {
                player: White,
                piece_type: King,
            } => 1,
            Piece {
                player: Black,
                piece_type: Man,
            } => 2,
            Piece {
                player: Black,
                piece_type: King,
            } => 3,
        }
    }

    pub fn unhash(hash: usize) -> Piece {
        assert_eq!(MAX_PIECE_HASH_VALUE, 3);
        assert!((0..=MAX_PIECE_HASH_VALUE).contains(&hash));
        match hash {
            0 => Piece {
                player: White,
                piece_type: Man,
            },
            1 => Piece {
                player: White,
                piece_type: King,
            },
            2 => Piece {
                player: Black,
                piece_type: Man,
            },
            3 => Piece {
                player: Black,
                piece_type: King,
            },
            _ => panic!(),
        }
    }

    pub fn is_white(self) -> bool {
        self.player.is_white()
    }

    pub fn is_black(self) -> bool {
        self.player.is_black()
    }

    pub fn is_man(self) -> bool {
        self.piece_type == Man
    }

    pub fn is_king(self) -> bool {
        self.piece_type == King
    }

    fn emoji(self) -> char {
        match (self.player, self.piece_type) {
            // As pieces are written on a black-themed terminal, colors are inverted
            (White, King) => '⛃',
            (White, Man) => '⛂',
            (Black, King) => '⛁',
            (Black, Man) => '⛀',
        }
    }
}

fn piece_opt_hash(piece: Option<Piece>) -> u64 {
    (if let Some(p) = piece {
        p.hash()
    } else {
        NONE_PIECE_HASH_VALUE
    }) as u64
}

impl Board {
    pub fn new() -> Board {
        let mut b = Board {
            board: [[None; BOARD_SIZE as usize]; BOARD_SIZE as usize],
            current_player: White,
            board_count: HashMap::new(),
            moves_without_capture: 0,
        };
        for y in 0..NB_PLAYERS_LINES {
            for x in 0..BOARD_SIZE {
                if is_playable(x, y) {
                    b.set(
                        x,
                        y,
                        Some(Piece {
                            player: White,
                            piece_type: Man,
                        }),
                    );
                }
            }
        }
        for y in (BOARD_SIZE - NB_PLAYERS_LINES)..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                if is_playable(x, y) {
                    b.set(
                        x,
                        y,
                        Some(Piece {
                            player: Black,
                            piece_type: Man,
                        }),
                    );
                }
            }
        }
        b.incr_board_count();
        b
    }

    pub fn hash(&self) -> BoardHash {
        fn hash_lines(board: &Board, start: i8, stop: i8) -> u64 {
            let mut res = 0;
            for y in start..stop {
                for x in 0..BOARD_SIZE {
                    if is_playable(x, y) {
                        res *= (MAX_OPT_PIECE_HASH_VALUE + 1) as u64;
                        res += piece_opt_hash(board.get(x, y));
                    }
                }
            }
            res
        }

        (
            self.get_player_is_white(),
            hash_lines(self, 0, NB_PLAYERS_LINES),
            hash_lines(self, NB_PLAYERS_LINES, BOARD_SIZE - NB_PLAYERS_LINES),
            hash_lines(self, BOARD_SIZE - NB_PLAYERS_LINES, BOARD_SIZE),
        )
    }

    pub fn get(&self, x: i8, y: i8) -> Option<Piece> {
        self.board[y as usize][x as usize]
    }

    fn set(&mut self, x: i8, y: i8, piece_opt: Option<Piece>) {
        assert!(is_playable(x, y));
        self.board[y as usize][x as usize] = piece_opt;
    }

    fn get_player(&self) -> Player {
        self.current_player
    }

    pub fn get_player_is_white(&self) -> bool {
        self.get_player() == White
    }

    fn switch_player(&mut self) {
        if self.get_player() == White {
            self.current_player = Black;
        } else {
            self.current_player = White;
        }
    }

    pub fn get_board_count(&self) -> i8 {
        match self.board_count.get(&self.hash()) {
            Some(&n) => n,
            None => 0,
        }
    }

    fn incr_board_count(&mut self) {
        self.board_count
            .insert(self.hash(), self.get_board_count() + 1);
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
        } else if self.possible_moves().is_empty() {
            Win(self.get_player().other())
        } else {
            Continue
        }
    }

    pub fn is_end_game(&self) -> bool {
        self.get_win_status() != Continue
    }

    pub fn get_pieces_counter(&self) -> Vec<i64> {
        let mut counter = vec![0; 4];
        for x in 0..BOARD_SIZE {
            for y in 0..BOARD_SIZE {
                if let Some(piece) = self.get(x, y) {
                    counter[piece.hash()] += 1;
                }
            }
        }
        counter
    }

    pub fn possible_moves(&self) -> Vec<Move> {
        if self.is_draw() {
            return Vec::new();
        }

        fn explore_moving(
            board: &mut Board,
            directions: &Vec<(i8, i8)>,
            x: i8,
            y: i8,
        ) -> Vec<Move> {
            let mut moves = Vec::new();
            for &(dx, dy) in directions {
                let (x2, y2) = (x + dx, y + dy);
                if is_playable(x2, y2) && board.get(x2, y2).is_none() {
                    moves.push(vec![(x2, y2)]);
                }
            }
            moves.iter_mut().for_each(|m| m.push((x, y)));
            moves
        }

        fn explore_jumping(
            board: &mut Board,
            directions: &Vec<(i8, i8)>,
            x: i8,
            y: i8,
        ) -> Vec<Move> {
            let mut moves = Vec::new();
            for &(dx, dy) in directions {
                let (x2, y2) = (x + dx, y + dy);
                let (x3, y3) = (x2 + dx, y2 + dy);
                if !(is_playable(x3, y3) && board.get(x3, y3).is_none()) {
                    continue;
                }
                if let Some(Piece {
                    player: p2,
                    piece_type: _,
                }) = board.get(x2, y2)
                {
                    if p2 != board.get_player() {
                        let jumping_piece = board.get(x, y);
                        let taken_piece = board.get(x2, y2);
                        board.set(x, y, None);
                        board.set(x2, y2, None);
                        board.set(x3, y3, jumping_piece);
                        let mut further_moves = explore_jumping(board, directions, x3, y3);
                        if further_moves.is_empty() {
                            further_moves.push(vec![(x3, y3)]);
                        }
                        moves.extend(further_moves);
                        board.set(x, y, jumping_piece);
                        board.set(x2, y2, taken_piece);
                        board.set(x3, y3, None);
                    }
                }
            }
            moves.iter_mut().for_each(|m| m.push((x, y)));
            moves
        }

        let mut board = self.clone();
        let mut moves = Vec::new();
        let mut mandatory_jump = false;

        for x in 0..BOARD_SIZE {
            for y in 0..BOARD_SIZE {
                let piece_opt = self.get(x, y);
                if piece_opt.is_none() || piece_opt.unwrap().player != board.get_player() {
                    continue;
                }
                let directions = allowed_directions(piece_opt.unwrap());
                if !mandatory_jump {
                    let moves_aux = explore_moving(&mut board, &directions, x, y);
                    moves.extend(moves_aux);
                }
                let moves_aux = explore_jumping(&mut board, &directions, x, y);
                if !mandatory_jump && !moves_aux.is_empty() {
                    moves = Vec::new();
                    mandatory_jump = true;
                }
                moves.extend(moves_aux);
            }
        }

        moves.iter_mut().for_each(|m| m.reverse());
        moves.sort();
        moves.iter_mut().for_each(|m| m.reverse());
        moves
    }

    pub fn play(&mut self, moves: &Move) {
        let possible_moves = self.possible_moves();
        if possible_moves.is_empty() {
            println!("Game Over!");
            if self.is_draw() {
                println!("> Draw");
            } else {
                println!("> Opponent won");
            }
            exit(0);
        }
        assert!(
            possible_moves.contains(moves),
            "Unauthorized move ({moves:?}). List: {possible_moves:?}"
        );
        let rev_moves = moves.iter().rev().copied().collect::<Vec<_>>();
        let (mut x, mut y) = rev_moves[0];
        let (x2, y2) = rev_moves[1];
        if i8::abs(y2 - y) == 1 {
            // No jump
            self.set(x2, y2, self.get(x, y));
            self.set(x, y, None);
            (x, y) = (x2, y2);
            self.incr_moves_without_capture();
        } else {
            for &(x2, y2) in &rev_moves[1..] {
                let (dx, dy) = ((x2 - x) / 2, (y2 - y) / 2);
                let (x_mid, y_mid) = (x + dx, y + dy);
                self.set(x_mid, y_mid, None);
                self.set(x2, y2, self.get(x, y));
                self.set(x, y, None);
                (x, y) = (x2, y2);
            }
            self.reset_moves_without_capture();
        }
        if y == BOARD_SIZE - 1
            && self.get(x, y).unwrap()
                == (Piece {
                    player: White,
                    piece_type: Man,
                })
        {
            self.set(
                x,
                y,
                Some(Piece {
                    player: White,
                    piece_type: King,
                }),
            );
            self.reset_moves_without_capture();
        } else if y == 0
            && self.get(x, y).unwrap()
                == (Piece {
                    player: Black,
                    piece_type: Man,
                })
        {
            self.set(
                x,
                y,
                Some(Piece {
                    player: Black,
                    piece_type: King,
                }),
            );
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
        let pieces_counter = self.get_pieces_counter();
        println!(
            "White: {} men, {} kings",
            pieces_counter[Piece {
                player: White,
                piece_type: Man
            }
            .hash()],
            pieces_counter[Piece {
                player: White,
                piece_type: King
            }
            .hash()]
        );
        println!(
            "Black: {} men, {} kings",
            pieces_counter[Piece {
                player: Black,
                piece_type: Man
            }
            .hash()],
            pieces_counter[Piece {
                player: Black,
                piece_type: King
            }
            .hash()]
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
                match self.get(x, y) {
                    Some(piece) => print!(" {} ", piece.emoji()),
                    None => {
                        if is_playable(x, y) {
                            print!(" • ");
                        } else {
                            print!("   ");
                        }
                    }
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

fn is_playable(x: i8, y: i8) -> bool {
    (0..BOARD_SIZE).contains(&x) && (0..BOARD_SIZE).contains(&y) && (x + y) % 2 == 0
}

fn allowed_directions(piece: Piece) -> Vec<(i8, i8)> {
    if piece.is_king() {
        vec![(1, 1), (-1, 1), (1, -1), (-1, -1)]
    } else if piece.is_white() {
        vec![(1, 1), (-1, 1)]
    } else {
        vec![(1, -1), (-1, -1)]
    }
}

pub fn char_of_x(x: i8) -> char {
    (x as u8 + 'A' as u8) as char
}

pub fn char_of_y(y: i8) -> char {
    (y as u8 + '1' as u8) as char
}
