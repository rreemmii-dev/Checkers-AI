use crate::checkers::player::Player;
use crate::checkers::win_status::WinStatus::{Continue, Draw, Win};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WinStatus {
    Win(Player),
    Draw,
    Continue,
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
