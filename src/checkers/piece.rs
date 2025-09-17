use crate::checkers::piece_type::PieceType;
use crate::checkers::piece_type::PieceType::{King, Man};
use crate::checkers::player::Player;
use crate::checkers::player::Player::{Black, White};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Piece {
    player: Player,
    piece_type: PieceType,
}

pub const PIECES: &[Piece] = &[
    Piece {
        player: White,
        piece_type: Man,
    },
    Piece {
        player: White,
        piece_type: King,
    },
    Piece {
        player: Black,
        piece_type: Man,
    },
    Piece {
        player: Black,
        piece_type: King,
    },
];

impl Piece {
    pub fn get_player(self) -> Player {
        self.player
    }

    pub fn get_piece_type(self) -> PieceType {
        self.piece_type
    }

    pub fn is_white(self) -> bool {
        self.player.is_white()
    }

    pub fn is_black(self) -> bool {
        self.player.is_black()
    }

    pub fn is_man(self) -> bool {
        self.piece_type.is_man()
    }

    pub fn is_king(self) -> bool {
        self.piece_type.is_king()
    }

    pub fn from(player: Player, piece_type: PieceType) -> Piece {
        Piece { player, piece_type }
    }

    pub fn emoji(self) -> char {
        match (self.player, self.piece_type) {
            // As pieces are written on a black-themed terminal, colors are inverted
            (White, King) => '⛃',
            (White, Man) => '⛂',
            (Black, King) => '⛁',
            (Black, Man) => '⛀',
        }
    }
}
