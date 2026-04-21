use crate::checkers::player::Player::{Black, White};
use crate::checkers::win_status::WinStatus::{Continue, Draw, Win};
use crate::neural_network::training::train::TournamentResult;
use plotpy::{Barplot, Plot};

struct GatheredTrainingResults {
    wins_white: Vec<u64>,
    draws_white: Vec<u64>,
    losses_white: Vec<u64>,
    wins_black: Vec<u64>,
    draws_black: Vec<u64>,
    losses_black: Vec<u64>,
}

pub fn display_results(tournament_result: &TournamentResult) {
    let gathered_results = gather_results(tournament_result);
    generate_graphs("graph_white.png", "graph_black.png", &gathered_results);
    print_best_neural_network(&gathered_results);
}

fn generate_graphs(
    file_name_white: &str,
    file_name_black: &str,
    gathered_results: &GatheredTrainingResults,
) {
    let GatheredTrainingResults {
        wins_white,
        draws_white,
        losses_white,
        wins_black,
        draws_black,
        losses_black,
    } = gathered_results;
    let nb_neural_networks = wins_white.len();

    generate_graph(
        file_name_white,
        "Win rate (playing white)",
        nb_neural_networks,
        vec![
            ("wins", wins_white, "green"),
            ("draws", draws_white, "blue"),
            ("losses", losses_white, "red"),
        ],
    );

    generate_graph(
        file_name_black,
        "Win rate (playing black)",
        nb_neural_networks,
        vec![
            ("wins", wins_black, "green"),
            ("draws", draws_black, "blue"),
            ("losses", losses_black, "red"),
        ],
    );
}

fn generate_graph(
    file_name: &str,
    title: &str,
    nb_neural_networks: usize,
    results: Vec<(&str, &Vec<u64>, &str)>,
) {
    let mut bar = Barplot::new();

    let mut bottom = vec![0.; nb_neural_networks];
    for (label, values, color) in results {
        bar.set_label(label)
            .set_bottom(&bottom)
            .set_extra("edgecolor='none'")
            .set_colors(&vec![color; nb_neural_networks])
            .draw(&(0..nb_neural_networks as u64).collect::<Vec<_>>(), values);
        for i in 0..values.len() {
            bottom[i] += values[i] as f64;
        }
    }

    let mut plot = Plot::new();
    plot.add(&bar)
        .set_title(title)
        .legend()
        .save(file_name)
        .unwrap();
    println!("Graph saved in: {}", file_name);
}

fn print_best_neural_network(gathered_results: &GatheredTrainingResults) {
    let GatheredTrainingResults {
        wins_white,
        draws_white,
        losses_white: _losses_white,
        wins_black,
        draws_black,
        losses_black: _losses_black,
    } = gathered_results;
    let nb_neural_networks = wins_white.len();

    let mut best_nn = 0;
    let mut best_score = 0;
    for nn in 0..nb_neural_networks {
        let score = 2 * (wins_white[nn] + wins_black[nn]) + draws_white[nn] + draws_black[nn];
        if score > best_score {
            best_nn = nn;
            best_score = score;
        }
    }
    let best_score = best_score / 2;
    let max_best_score = 2 * nb_neural_networks;
    println!(
        "Best score: {} / {} for neural network ID {}",
        best_score, max_best_score, best_nn
    );
}

fn gather_results(tournament_result: &TournamentResult) -> GatheredTrainingResults {
    let nb_neural_networks = tournament_result.len();
    let mut wins_white = vec![0; nb_neural_networks];
    let mut draws_white = vec![0; nb_neural_networks];
    let mut losses_white = vec![0; nb_neural_networks];
    let mut wins_black = vec![0; nb_neural_networks];
    let mut draws_black = vec![0; nb_neural_networks];
    let mut losses_black = vec![0; nb_neural_networks];
    for (white_id, results) in tournament_result.iter().enumerate() {
        for (black_id, &result) in results.iter().enumerate() {
            match result {
                Win(White) => {
                    wins_white[white_id] += 1;
                    losses_black[black_id] += 1;
                }
                Win(Black) => {
                    losses_white[white_id] += 1;
                    wins_black[black_id] += 1;
                }
                Draw => {
                    draws_white[white_id] += 1;
                    draws_black[black_id] += 1;
                }
                Continue => panic!("Continue"),
            }
        }
    }
    GatheredTrainingResults {
        wins_white,
        draws_white,
        losses_white,
        wins_black,
        draws_black,
        losses_black,
    }
}
