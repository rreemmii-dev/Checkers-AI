// For more information about bitboards, see the subsection #Performance#Bitboards in the README.md

use crate::checkers::board::{BOARD_SIZE, char_of_x, char_of_y, is_playable};

pub trait BitBoard {
    fn set_bit(&mut self, n: usize, value: bool);
    fn get_bit(self, n: usize) -> bool;
    fn set(&mut self, x: i8, y: i8, value: bool);
    fn is_some(self, x: i8, y: i8) -> bool;
    fn is_none(self, x: i8, y: i8) -> bool;
    fn move_direction(self, dir: &(i8, i8)) -> Self;
    fn display(self);
}

const EVEN_ROW_MASK: u32 = 0x0f0f0f0f; // Mask for y = 0, 2, 4, 6
const ODD_ROW_MASK: u32 = !EVEN_ROW_MASK; // Mask for y = 1, 3, 5, 7
const LEFT_COLUMN_MASK: u32 = 0x01010101; // Mask for x = 0
const RIGHT_COLUMN_MASK: u32 = 0x80808080; // Mask for x = 7

impl BitBoard for u32 {
    fn set_bit(&mut self, n: usize, value: bool) {
        if value {
            let mask = 1 << n;
            *self |= mask;
        } else {
            let mask = !(1 << n);
            *self &= mask;
        }
    }

    fn get_bit(self, n: usize) -> bool {
        let mask = 1 << n;
        self & mask != 0
    }

    fn set(&mut self, x: i8, y: i8, value: bool) {
        self.set_bit(bitboard_index(x, y), value);
    }

    fn is_some(self, x: i8, y: i8) -> bool {
        self.get_bit(bitboard_index(x, y))
    }

    fn is_none(self, x: i8, y: i8) -> bool {
        !self.is_some(x, y)
    }

    fn move_direction(self, direction: &(i8, i8)) -> Self {
        if direction == &(1, 1) {
            // North East
            ((EVEN_ROW_MASK & self) << 4) | ((ODD_ROW_MASK & !RIGHT_COLUMN_MASK & self) << 5)
        } else if direction == &(-1, 1) {
            // North West
            ((EVEN_ROW_MASK & !LEFT_COLUMN_MASK & self) << 3) | ((ODD_ROW_MASK & self) << 4)
        } else if direction == &(1, -1) {
            // South East
            ((EVEN_ROW_MASK & self) >> 4) | ((ODD_ROW_MASK & !RIGHT_COLUMN_MASK & self) >> 3)
        } else if direction == &(-1, -1) {
            // South West
            ((EVEN_ROW_MASK & !LEFT_COLUMN_MASK & self) >> 5) | ((ODD_ROW_MASK & self) >> 4)
        } else {
            panic!()
        }
    }

    fn display(self) {
        print!("   ");
        for x in 0..BOARD_SIZE {
            print!(" {} ", char_of_x(x));
        }
        println!();
        for y in (0..BOARD_SIZE).rev() {
            print!(" {} ", char_of_y(y));
            for x in 0..BOARD_SIZE {
                if is_playable(x, y) {
                    if self.is_some(x, y) {
                        print!(" X ");
                    } else {
                        print!(" • ");
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

fn bitboard_index(x: i8, y: i8) -> usize {
    assert!(is_playable(x, y));
    (y * BOARD_SIZE / 2 + x / 2) as usize
}
