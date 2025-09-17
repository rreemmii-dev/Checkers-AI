use crate::checkers::board::{Board, BoardHash, Move};
use crate::score::{NEG_INFINITY, POS_INFINITY};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

// pub static NB_NODES: AtomicU64 = AtomicU64::new(0);

/// heuristic_score: Current player point of view

pub fn alpha_beta_score(
    board: &Board,
    alpha: i64,
    beta: i64,
    heuristic_score: &dyn Fn(&Board) -> i64,
    cache: &mut HashMap<(BoardHash, i8), (i64, i64)>,
    best_move_first_min_depth: i8,
    best_move_first_skip_size: i8,
    depth: i8,
    cancel_search: &AtomicBool,
) -> i64 {
    if cancel_search.load(Ordering::Acquire) {
        return 0;
    }

    // NB_NODES.fetch_add(1, Ordering::AcqRel);

    // ********** Import cache results **********
    let alpha_init = alpha;
    let beta_init = beta;
    let mut alpha = alpha;
    let mut beta = beta;
    if let Some(&(min, max)) = cache.get(&(board.hash(), depth)) {
        if max <= alpha_init {
            return alpha_init;
        } else if min >= beta_init {
            return beta_init;
        } else if min == max {
            return min;
        }
        alpha = i64::max(alpha, min);
        beta = i64::min(beta, max);
    }

    // ********** Main: get score value **********
    let score = 'score: {
        if depth == 0 || board.is_end_game() {
            break 'score heuristic_score(board);
        }

        // ********** Choose best move first **********
        let moves = if depth >= best_move_first_min_depth {
            let mut moves = board
                .possible_moves()
                .into_iter()
                .map(|m| {
                    let mut cloned_board = board.clone();
                    cloned_board.play(&m);
                    let res = if best_move_first_skip_size % 2 == 0 {
                        1
                    } else {
                        -1
                    } * alpha_beta_score(
                        &cloned_board,
                        NEG_INFINITY,
                        POS_INFINITY,
                        heuristic_score,
                        cache,
                        best_move_first_min_depth,
                        best_move_first_skip_size,
                        depth - best_move_first_skip_size,
                        cancel_search,
                    );
                    (m, res)
                })
                .collect::<Vec<_>>();
            moves.sort_by(|a, b| b.1.cmp(&a.1));
            moves.into_iter().map(|(m, _)| m).collect::<Vec<_>>()
        } else {
            board.possible_moves()
        };

        let mut alpha = alpha;
        for m in moves {
            let mut cloned_board = board.clone();
            cloned_board.play(&m);
            let res = -alpha_beta_score(
                &cloned_board,
                -beta,
                -alpha,
                heuristic_score,
                cache,
                best_move_first_min_depth,
                best_move_first_skip_size,
                depth - 1,
                cancel_search,
            );

            // ********** Alpha-beta pruning **********
            if res >= beta {
                break 'score beta;
            } else if res > alpha {
                alpha = res;
            }
        }
        alpha
    };

    if cancel_search.load(Ordering::Acquire) {
        return 0;
    }

    // ********** Store results **********
    cache.insert((board.hash(), depth), {
        if let Some(&(min, max)) = cache.get(&(board.hash(), depth)) {
            if score <= alpha_init {
                (min, i64::min(max, alpha_init))
            } else if score >= beta_init {
                (i64::max(min, beta_init), max)
            } else {
                (score, score)
            }
        } else {
            if score <= alpha_init {
                (NEG_INFINITY, alpha_init)
            } else if score >= beta_init {
                (beta_init, POS_INFINITY)
            } else {
                (score, score)
            }
        }
    });

    cache[&(board.hash(), depth)].0
}

pub fn alpha_beta_list(
    board: &Board,
    heuristic_score: &dyn Fn(&Board) -> i64,
    best_move_first_min_depth: i8,
    best_move_first_skip_size: i8,
    depth: i8,
    cancel_search: &AtomicBool,
) -> Option<(i64, Vec<Move>)> {
    if cancel_search.load(Ordering::Acquire) {
        return None;
    }

    let mut cache = HashMap::new();

    if depth == 0 || board.is_end_game() {
        return Some((heuristic_score(board), Vec::new()));
    }

    // ********** Choose best move first **********
    let moves = if depth >= best_move_first_min_depth {
        let mut moves = board
            .possible_moves()
            .into_iter()
            .map(|m| {
                let mut cloned_board = board.clone();
                cloned_board.play(&m);
                let res = if best_move_first_skip_size % 2 == 0 {
                    1
                } else {
                    -1
                } * alpha_beta_score(
                    &cloned_board,
                    NEG_INFINITY,
                    POS_INFINITY,
                    heuristic_score,
                    &mut cache,
                    best_move_first_min_depth,
                    best_move_first_skip_size,
                    depth - best_move_first_skip_size,
                    cancel_search,
                );
                (m, res)
            })
            .collect::<Vec<_>>();
        moves.sort_by(|a, b| b.1.cmp(&a.1));
        moves.into_iter().map(|(m, _)| m).collect::<Vec<_>>()
    } else {
        board.possible_moves()
    };

    let mut best_score = NEG_INFINITY;
    let mut best_moves = Vec::new();
    for m in moves {
        let mut cloned_board = board.clone();
        cloned_board.play(&m);
        let alpha = best_score - 1;
        let beta = POS_INFINITY;
        let res = -alpha_beta_score(
            &cloned_board,
            -beta,
            -alpha,
            heuristic_score,
            &mut cache,
            best_move_first_min_depth,
            best_move_first_skip_size,
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

    Some((best_score, best_moves))
}

fn threaded_score(
    board: Board,
    heuristic_score: Arc<dyn Fn(&Board) -> i64 + Send + Sync>,
    best_move_first_min_depth: i8,
    best_move_first_skip_size: i8,
    depth: i8,
    threads_depth: i8,
    cancel_search: Arc<AtomicBool>,
) -> i64 {
    if cancel_search.load(Ordering::Acquire) {
        return 0;
    }

    // NB_NODES.fetch_add(1, Ordering::AcqRel);

    let mut cache = HashMap::new();

    if threads_depth == 0 {
        return alpha_beta_score(
            &board,
            NEG_INFINITY,
            POS_INFINITY,
            &*heuristic_score,
            &mut cache,
            best_move_first_min_depth,
            best_move_first_skip_size,
            depth,
            &*cancel_search,
        );
    }
    if depth == 0 || board.is_end_game() {
        return heuristic_score(&board);
    }
    let mut handle = Vec::new();
    let board = Arc::new(board);
    for m in board.possible_moves() {
        let board = board.clone();
        let heuristic_score = heuristic_score.clone();
        let cancel_search = cancel_search.clone();
        handle.push(thread::spawn(move || {
            let mut cloned_board = (*board).clone();
            cloned_board.play(&m);
            -threaded_score(
                cloned_board,
                heuristic_score,
                best_move_first_min_depth,
                best_move_first_skip_size,
                depth - 1,
                threads_depth - 1,
                cancel_search,
            )
        }));
    }
    let mut best_score = NEG_INFINITY;
    for h in handle {
        let score = h.join().unwrap();
        if score > best_score {
            best_score = score;
        }
    }

    best_score
}

pub fn threaded_moves_list(
    board: Board,
    heuristic_score: Arc<dyn Fn(&Board) -> i64 + Send + Sync>,
    best_move_first_min_depth: i8,
    best_move_first_skip_size: i8,
    depth: i8,
    threads_depth: i8,
    cancel_search: Arc<AtomicBool>,
) -> Option<(i64, Vec<Move>)> {
    if cancel_search.load(Ordering::Acquire) {
        return None;
    }

    if depth == 0 || board.is_end_game() {
        return Some((heuristic_score(&board), Vec::new()));
    }
    let mut handle = Vec::new();
    let board = Arc::new(board);
    for m in board.possible_moves() {
        let board = board.clone();
        let heuristic_score = heuristic_score.clone();
        let cancel_search = cancel_search.clone();
        handle.push(thread::spawn(move || {
            let mut cloned_board = (*board).clone();
            cloned_board.play(&m);
            let res = -threaded_score(
                cloned_board,
                heuristic_score,
                best_move_first_min_depth,
                best_move_first_skip_size,
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

    Some((best_score, best_moves))
}
