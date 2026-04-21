use crate::checkers::board::{Board, BoardHash, Move};
use crate::players::alpha_beta::score::{NEG_INFINITY, POS_INFINITY};
use std::cmp::Reverse;
use std::collections::HashMap;
#[cfg(not(nn_is_sync))]
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

// heuristic_score: Current player point of view

const MAX_THREADING_DEPTH: i8 = 2; // recommended: 1 or 2 (branch-size usually between 5 and 10)
const BEST_MOVE_FIRST_MIN_DEPTH: i8 = 2 * 4; // use "best move first" strategy if depth >= BEST_MOVE_FIRST_MIN_DEPTH
const BEST_MOVE_FIRST_SKIP_SIZE: i8 = 2 * 3; // choose the best move by exploring at depth := depth - BEST_MOVE_FIRST_SKIP_SIZE

pub fn alpha_beta_moves_list(
    board: &Board,
    // TODO: Use a pub type `Arc<...>` everywhere when being part of stable Rust
    #[cfg(nn_is_sync)] heuristic_score: Arc<impl Fn(&Board) -> i64 + Send + Sync + 'static>,
    #[cfg(not(nn_is_sync))] heuristic_score: Arc<impl Fn(&Board) -> i64 + Send + Clone + 'static>,
    depth: i8,
    threaded: bool,
    cancel_search: Arc<AtomicBool>,
) -> Option<Vec<Move>> {
    if threaded {
        threaded_best_moves(
            board.to_owned(),
            heuristic_score,
            depth,
            MAX_THREADING_DEPTH,
            cancel_search,
        )
    } else {
        alpha_beta_best_moves(board, &*heuristic_score, depth, &cancel_search)
    }
}

fn alpha_beta_score(
    board: &Board,
    alpha: i64,
    beta: i64,
    heuristic_score: &impl Fn(&Board) -> i64,
    cache: &mut HashMap<(BoardHash, i8), (i64, i64)>,
    depth: i8,
    cancel_search: &AtomicBool,
) -> i64 {
    if cancel_search.load(Ordering::Acquire) {
        return 0;
    }
    if depth == 0 || board.is_end_game() {
        return heuristic_score(board);
    }

    // ********** Import cache results **********
    let (mut alpha, mut beta) = (alpha, beta);
    if let Some(&(min, max)) = cache.get(&(board.hash(), depth)) {
        if max <= alpha {
            return alpha;
        }
        if min >= beta {
            return beta;
        }
        if min == max {
            return min;
        }
        alpha = i64::max(alpha, min);
        beta = i64::min(beta, max);
    }
    let (alpha_init, beta_init) = (alpha, beta);

    // ********** Main: get score value **********
    let moves = best_move_first(board, heuristic_score, cache, depth, cancel_search);
    let mut score = alpha;
    for m in moves {
        let mut cloned_board = board.clone();
        cloned_board.play(&m);
        let res = -alpha_beta_score(
            &cloned_board,
            -beta,
            -alpha,
            heuristic_score,
            cache,
            depth - 1,
            cancel_search,
        );

        // ********** Alpha-beta pruning **********
        if res >= beta {
            score = beta;
            break;
        }
        alpha = i64::max(alpha, res);
        score = alpha;
    }

    if cancel_search.load(Ordering::Acquire) {
        return 0;
    }

    // ********** Store results **********
    cache.insert((board.hash(), depth), {
        let (stored_min, stored_max) = cache
            .get(&(board.hash(), depth))
            .copied()
            .unwrap_or((NEG_INFINITY, POS_INFINITY));
        if score <= alpha_init {
            (stored_min, i64::min(stored_max, alpha_init))
        } else if score >= beta_init {
            (i64::max(stored_min, beta_init), stored_max)
        } else {
            (score, score)
        }
    });

    score
}

fn alpha_beta_best_moves(
    board: &Board,
    heuristic_score: &impl Fn(&Board) -> i64,
    depth: i8,
    cancel_search: &AtomicBool,
) -> Option<Vec<Move>> {
    if cancel_search.load(Ordering::Acquire) {
        return None;
    }

    if depth == 0 || board.is_end_game() {
        return Some(Vec::new());
    }

    let mut cache = HashMap::new();

    let moves = best_move_first(board, heuristic_score, &mut cache, depth, cancel_search);

    let mut best_score = NEG_INFINITY;
    let mut best_moves = Vec::new();
    for m in moves {
        let mut cloned_board = board.clone();
        cloned_board.play(&m);
        let alpha = best_score;
        let beta = POS_INFINITY;
        let res = -alpha_beta_score(
            &cloned_board,
            -beta,
            -alpha,
            heuristic_score,
            &mut cache,
            depth - 1,
            cancel_search,
        );
        if res > best_score {
            best_score = res;
            best_moves = vec![m.clone()];
        } else if res == best_score {
            best_moves.push(m.clone());
        }
    }

    if cancel_search.load(Ordering::Acquire) {
        return None;
    }

    Some(best_moves)
}

fn threaded_score(
    board: Board,
    #[cfg(nn_is_sync)] heuristic_score: Arc<impl Fn(&Board) -> i64 + Send + Sync + 'static>,
    #[cfg(not(nn_is_sync))] heuristic_score: Arc<impl Fn(&Board) -> i64 + Send + Clone + 'static>,
    depth: i8,
    threads_depth: i8,
    cancel_search: Arc<AtomicBool>,
) -> i64 {
    if cancel_search.load(Ordering::Acquire) {
        return 0;
    }

    let mut cache = HashMap::new();

    if threads_depth == 0 {
        return alpha_beta_score(
            &board,
            NEG_INFINITY,
            POS_INFINITY,
            &*heuristic_score,
            &mut cache,
            depth,
            &cancel_search,
        );
    }
    if depth == 0 || board.is_end_game() {
        return heuristic_score(&board);
    }
    let mut handle = Vec::new();
    let board = Arc::new(board);
    for m in board.possible_moves() {
        let board = board.clone();
        let heuristic_score = cfg_select! {
            nn_is_sync => heuristic_score.clone(),
            not(nn_is_sync) => heuristic_score.deref().to_owned(),
        };
        let cancel_search = cancel_search.clone();
        handle.push(thread::spawn(move || {
            let mut cloned_board = (*board).clone();
            cloned_board.play(&m);
            -threaded_score(
                cloned_board,
                cfg_select! {
                    nn_is_sync => heuristic_score,
                    not(nn_is_sync) => Arc::new(heuristic_score)
                },
                depth - 1,
                threads_depth - 1,
                cancel_search,
            )
        }));
    }
    let mut best_score = NEG_INFINITY;
    for h in handle {
        let score = h.join().unwrap();
        best_score = i64::max(best_score, score);
    }

    best_score
}

fn threaded_best_moves(
    board: Board,
    #[cfg(nn_is_sync)] heuristic_score: Arc<impl Fn(&Board) -> i64 + Send + Sync + 'static>,
    #[cfg(not(nn_is_sync))] heuristic_score: Arc<impl Fn(&Board) -> i64 + Send + Clone + 'static>,
    depth: i8,
    threads_depth: i8,
    cancel_search: Arc<AtomicBool>,
) -> Option<Vec<Move>> {
    if cancel_search.load(Ordering::Acquire) {
        return None;
    }

    if depth == 0 || board.is_end_game() {
        return Some(Vec::new());
    }

    let mut handle = Vec::new();
    let board = Arc::new(board);
    for m in board.possible_moves() {
        let board = board.clone();
        let heuristic_score = cfg_select! {
            nn_is_sync => heuristic_score.clone(),
            not(nn_is_sync) => heuristic_score.deref().to_owned(),
        };
        let cancel_search = cancel_search.clone();
        handle.push(thread::spawn(move || {
            let mut cloned_board = (*board).clone();
            cloned_board.play(&m);
            let res = -threaded_score(
                cloned_board,
                cfg_select! {
                    nn_is_sync => heuristic_score,
                    not(nn_is_sync) => Arc::new(heuristic_score)
                },
                depth - 1,
                threads_depth - 1,
                cancel_search,
            );
            (res, m)
        }));
    }
    let mut best_score = NEG_INFINITY;
    let mut best_moves = Vec::new();
    for h in handle {
        let (score, m) = h.join().unwrap();
        if score > best_score {
            best_score = score;
            best_moves = vec![m];
        } else if score == best_score {
            best_moves.push(m);
        }
    }

    if cancel_search.load(Ordering::Acquire) {
        return None;
    }

    Some(best_moves)
}

fn best_move_first(
    board: &Board,
    heuristic_score: &impl Fn(&Board) -> i64,
    cache: &mut HashMap<(BoardHash, i8), (i64, i64)>,
    depth: i8,
    cancel_search: &AtomicBool,
) -> Vec<Move> {
    if depth < BEST_MOVE_FIRST_MIN_DEPTH {
        return board.possible_moves();
    }
    let mut moves = board
        .possible_moves()
        .into_iter()
        .map(|m| {
            let mut cloned_board = board.clone();
            cloned_board.play(&m);
            let sign = if BEST_MOVE_FIRST_SKIP_SIZE % 2 == 0 {
                1
            } else {
                -1
            };
            let score = alpha_beta_score(
                &cloned_board,
                NEG_INFINITY,
                POS_INFINITY,
                heuristic_score,
                cache,
                depth - BEST_MOVE_FIRST_SKIP_SIZE,
                cancel_search,
            );
            let res = sign * score;
            (m, res)
        })
        .collect::<Vec<_>>();
    moves.sort_by_key(|m| Reverse(m.1));
    moves.into_iter().map(|(m, _)| m).collect::<Vec<_>>()
}
