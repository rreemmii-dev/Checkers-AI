use crate::checkers::piece_type::PieceType::{King, Man};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PieceType {
    Man,
    King,
}

impl PieceType {
    pub fn is_man(self) -> bool {
        self == Man
    }

    pub fn is_king(self) -> bool {
        self == King
    }
}
