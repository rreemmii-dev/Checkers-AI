use crate::checkers::player::Player;
use crate::checkers::win_status::WinStatus::Continue;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WinStatus {
    Win(Player),
    Draw,
    Continue,
}

impl WinStatus {
    pub fn is_end_game(self) -> bool {
        self != Continue
    }
}
