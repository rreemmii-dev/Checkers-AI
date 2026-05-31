use crate::consts::{NeuralNetwork, NeuralNetworkFloat, TOURNAMENT_STEP_SIZE};
use crate::neural_network::neural_network::{
    NeuralNetworkParameters, NeuralNetworkTrait, generate_neural_networks, get_nodes_per_layer,
};
use crate::neural_network::types::matrix::Matrix;
use std::fs;
use std::fs::{File, create_dir, read_dir};
use std::io::Write;
use std::path::Path;

// Neural network number 3 from iteration 42 is stored in $FOLDER/42/neural_network_3.txt

pub fn store_new_neural_networks(neural_networks: &[NeuralNetwork], folder: &str) {
    let folder_id = get_latest_folder_id(folder) + 1;
    let new_folder_path = format!("{folder}/{folder_id}");
    create_dir(Path::new(&new_folder_path)).unwrap();
    store_neural_networks(neural_networks, &new_folder_path);
}

pub fn load_latest_neural_networks(folder: &str) -> Vec<NeuralNetwork> {
    let folder_id = get_latest_folder_id(folder);
    load_neural_networks(&format!("{folder}/{folder_id}"))
}

pub fn load_all_neural_networks(folder: &str) -> Vec<NeuralNetwork> {
    let mut neural_networks = Vec::new();
    let nb_folders = read_dir(folder).unwrap().count();
    for folder_id in 0..nb_folders {
        for file in read_dir(format!("{folder}/{folder_id}"))
            .unwrap()
            .map(Result::unwrap)
        {
            if file.file_name() != "neural_network_0.txt" || folder_id % TOURNAMENT_STEP_SIZE != 0 {
                continue;
            }
            println!("Loads: {}", file.path().display());
            neural_networks.push(load_neural_network(file.path().to_str().unwrap()));
        }
    }
    neural_networks
}

pub fn load_neural_network(file: &str) -> NeuralNetwork {
    NeuralNetwork::import(&load_parameters(file))
}

fn store_neural_networks(neural_networks: &[NeuralNetwork], folder: &str) {
    for (id, neural_network) in neural_networks.iter().enumerate() {
        store_neural_network(neural_network, &format!("{folder}/neural_network_{id}.txt"));
    }
}

fn load_neural_networks(folder: &str) -> Vec<NeuralNetwork> {
    let mut files_in_folder = read_dir(folder).unwrap().peekable();
    if files_in_folder.peek().is_none() {
        // Empty folder
        let neural_networks = generate_neural_networks();
        store_neural_networks(&neural_networks, folder);
        return neural_networks;
    }
    let mut neural_networks = Vec::new();
    for file in files_in_folder.map(Result::unwrap) {
        let new_neural_network = load_neural_network(file.path().to_str().unwrap());
        neural_networks.push(new_neural_network);
    }
    neural_networks
}

fn store_neural_network(neural_network: &NeuralNetwork, file: &str) {
    store_parameters(&neural_network.export(), file);
}

fn get_latest_folder_id(folder: &str) -> usize {
    if read_dir(folder).unwrap().count() == 0 {
        let default_folder_name = folder.to_string() + "/0";
        create_dir(&default_folder_name).unwrap();
        return 0;
    }
    let latest_folder = read_dir(folder)
        .unwrap()
        .map(Result::unwrap)
        .max_by(|f1, f2| {
            f1.metadata()
                .unwrap()
                .modified()
                .unwrap()
                .cmp(&f2.metadata().unwrap().modified().unwrap())
        })
        .unwrap()
        .path();
    latest_folder
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .parse::<usize>()
        .unwrap()
}

fn store_parameters(parameters: &NeuralNetworkParameters, path: &str) {
    let mut file = File::create(path).unwrap();
    let mut content = String::new();

    content += &store_header(get_nodes_per_layer(parameters), parameters.learning_rate);

    for l in 0..parameters.weights.len() {
        content += "\n\n";
        content += &store_matrix(&parameters.weights[l]);
        content += "\n\n";
        content += &store_matrix(&parameters.biases[l]);
    }

    file.write_all(content.as_bytes()).unwrap();
}

fn load_parameters(path: &str) -> NeuralNetworkParameters {
    let content = fs::read_to_string(path).unwrap();
    let mut blocs = content.split("\n\n");

    let (nodes_per_layer, learning_rate) = load_header(blocs.next().unwrap());

    let mut weights = Vec::new();
    let mut biases = Vec::new();
    for (id, matrix) in blocs.enumerate() {
        if id % 2 == 0 {
            weights.push(load_matrix(matrix));
        } else {
            biases.push(load_matrix(matrix));
        }
    }

    let parameters = NeuralNetworkParameters {
        learning_rate,
        weights,
        biases,
    };
    assert_eq!(nodes_per_layer, get_nodes_per_layer(&parameters));
    parameters
}

fn store_matrix(matrix: &Matrix) -> String {
    let mut res = String::new();
    for x in 0..matrix.height() {
        for y in 0..matrix.width() {
            res += &(matrix.get(x, y) as f64).to_string();
            res += " ";
        }
        res.pop(); // Removes trailing space
        res += "\n";
    }
    res.pop(); // Removes trailing new line
    res
}

fn load_matrix(text: &str) -> Matrix {
    if text.is_empty() {
        return Matrix::new(0, 0);
    }
    let height = text.lines().count();
    let width = text.lines().next().unwrap().split(' ').count();
    let mut res = Matrix::new(height, width);
    for (x, line) in text.lines().enumerate() {
        for (y, value) in line.split(' ').enumerate() {
            res.set(x, y, value.parse::<f64>().unwrap() as NeuralNetworkFloat);
        }
    }
    res
}

fn store_header(nodes_per_layer: Vec<usize>, learning_rate: NeuralNetworkFloat) -> String {
    let mut res = String::new();
    for x in nodes_per_layer {
        res += &x.to_string();
        res += " ";
    }
    res.pop(); // Removes trailing space
    res += "\n";
    res += &learning_rate.to_string();
    res
}

fn load_header(text: &str) -> (Vec<usize>, NeuralNetworkFloat) {
    let mut header_lines = text.lines();
    let nodes_per_layer = header_lines.next().unwrap();
    let nodes_per_layer = nodes_per_layer
        .split(' ')
        .map(|x| x.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    let learning_rate = header_lines.next().unwrap().parse::<f64>().unwrap() as NeuralNetworkFloat;
    assert!(header_lines.next().is_none());
    (nodes_per_layer, learning_rate)
}
