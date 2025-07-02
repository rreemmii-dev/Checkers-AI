use crate::checkers::Player::{Black, White};
use crate::checkers::WinStatus::{Draw, Win};
use crate::checkers::{
    BOARD_SIZE, Board, MAX_BOARD_COUNT, MAX_MOVES_WITHOUT_CAPTURE, NB_PLAYERS_LINES, Piece,
};

const MAN_SCORE: i64 = 1;
const KING_SCORE: i64 = 2;
pub const WHITE_SIGN: i64 = 1;
pub const BLACK_SIGN: i64 = -1;
const MAX_BOARD_COUNT_SCORE_COEF: i64 = 100;
const MAX_MOVES_WITHOUT_CAPTURE_SCORE_COEF: i64 = 100;
const MAX_SCORE_COEF: i64 = MAX_BOARD_COUNT_SCORE_COEF * MAX_MOVES_WITHOUT_CAPTURE_SCORE_COEF;
const MAX_SCORE_WITHOUT_COEF: i64 = (NB_PLAYERS_LINES * BOARD_SIZE / 2) as i64 * KING_SCORE;
pub const POS_INFINITY: i64 = MAX_SCORE_WITHOUT_COEF * MAX_SCORE_COEF + 1;
pub const NEG_INFINITY: i64 = -POS_INFINITY;
const WHITE_WIN: i64 = WHITE_SIGN * (POS_INFINITY - 1);
const BLACK_WIN: i64 = BLACK_SIGN * (POS_INFINITY - 1);
const DRAW: i64 = 0;

fn piece_score(piece: Piece) -> i64 {
    let sign = if piece.is_white() {
        WHITE_SIGN
    } else {
        BLACK_SIGN
    };
    let value = if piece.is_king() {
        KING_SCORE
    } else {
        MAN_SCORE
    };
    sign * value
}

pub fn naive_score(board: &Board) -> i64 {
    match board.get_win_status() {
        Draw => return DRAW,
        Win(White) => return WHITE_WIN,
        Win(Black) => return BLACK_WIN,
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
    for (piece_hash, count) in board.get_pieces_counter().into_iter().enumerate() {
        score += count * piece_score(Piece::unhash(piece_hash));
    }
    score *= coef_board_count(board.get_board_count());
    score *= coef_moves_without_capture(board.get_moves_without_capture());
    score
}
