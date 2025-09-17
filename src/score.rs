use crate::checkers::board::{
    BOARD_SIZE, Board, MAX_BOARD_COUNT, MAX_MOVES_WITHOUT_CAPTURE, NB_PLAYERS_LINES, is_playable,
};
use crate::checkers::piece::Piece;
use crate::checkers::player::Player;
use crate::checkers::win_status::WinStatus::{Draw, Win};

const MAX_PIECE_SCORE: i64 = 1000;
pub const WHITE_SIGN: i64 = 1;
pub const BLACK_SIGN: i64 = -1;
const MAX_BOARD_COUNT_SCORE_COEF: i64 = 100;
const MAX_MOVES_WITHOUT_CAPTURE_SCORE_COEF: i64 = 100;
const MAX_SCORE_COEF: i64 = MAX_BOARD_COUNT_SCORE_COEF * MAX_MOVES_WITHOUT_CAPTURE_SCORE_COEF;
const MAX_SCORE_WITHOUT_COEF: i64 = (NB_PLAYERS_LINES * BOARD_SIZE / 2) as i64 * MAX_PIECE_SCORE;
pub const POS_INFINITY: i64 = MAX_SCORE_WITHOUT_COEF * MAX_SCORE_COEF + 1;
pub const NEG_INFINITY: i64 = -POS_INFINITY;
const WHITE_WIN: i64 = WHITE_SIGN * (POS_INFINITY - 1);
const BLACK_WIN: i64 = BLACK_SIGN * (POS_INFINITY - 1);
const DRAW: i64 = 0;

fn piece_score(piece: Piece, x: i8, y: i8) -> i64 {
    let sign = if piece.is_white() {
        WHITE_SIGN
    } else {
        BLACK_SIGN
    };
    let y = if piece.is_white() {
        y
    } else {
        BOARD_SIZE - 1 - y
    };
    let value = if piece.is_king() {
        let y_dist = i8::min(BOARD_SIZE - 1 - y, y);
        let x_dist = i8::min(BOARD_SIZE - 1 - x, x);
        let tot_dist = x_dist + y_dist;
        match tot_dist {
            0 => 200,
            1 => 210,
            2 => 225,
            3 => 250,
            4 => 275,
            5 => 290,
            6 => 300,
            _ => panic!("{}", tot_dist),
        }
    } else {
        match y {
            0..=2 => 100,
            3 => 105,
            4 => 112,
            5 => 125,
            6 => 150,
            _ => panic!("{}", y),
        }
    };
    assert!(value < MAX_PIECE_SCORE);
    sign * value
}

pub fn naive_score(board: &Board) -> i64 {
    match board.get_win_status() {
        Draw => return DRAW,
        Win(Player::White) => return WHITE_WIN,
        Win(Player::Black) => return BLACK_WIN,
        _ => (),
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
    for x in 0..BOARD_SIZE {
        for y in 0..BOARD_SIZE {
            if is_playable(x, y)
                && let Some(piece) = board.get(x, y)
            {
                score += piece_score(piece, x, y);
            }
        }
    }

    score *= coef_board_count(board.get_board_count());
    score *= coef_moves_without_capture(board.get_moves_without_capture());
    score
}
