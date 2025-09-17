use crate::checkers::player::Player::{Black, White};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Player {
    White,
    Black,
}

impl Player {
    pub fn is_white(self) -> bool {
        self == White
    }

    pub fn is_black(self) -> bool {
        self == Black
    }

    pub fn other(self) -> Player {
        match self {
            White => Black,
            Black => White,
        }
    }
}
