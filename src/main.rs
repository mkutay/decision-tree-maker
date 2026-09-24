use std::collections::{HashMap, HashSet};

enum DecisionTree {
    Leaf(String),
    Branch {
        column: usize,
        default_class: String,
        branches: HashMap<String, Box<DecisionTree>>,
    },
}

fn parse_data(content: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let mut lines = content.lines().filter(|line| !line.trim().is_empty());

    let features: Vec<String> = lines
        .next()
        .map(|header| header.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let rows: Vec<Vec<String>> = lines
        .map(|line| line.split(',').map(|s| s.trim().to_string()).collect())
        .collect();

    (features, rows)
}

fn column_labels(rows: &[Vec<String>], column: usize) -> HashSet<&str> {
    rows.iter().map(|row| row[column].as_str()).collect()
}

fn entropy(rows: &[Vec<String>], target_column: usize) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for row in rows {
        *counts.entry(&row[target_column]).or_default() += 1;
    }

    counts
        .values()
        .map(|&count| {
            let p = (count as f64) / (rows.len() as f64);
            -p * p.log2()
        })
        .sum()
}

fn groups(rows: &[Vec<String>], column: usize) -> HashMap<String, Vec<Vec<String>>> {
    let mut groups: HashMap<String, Vec<Vec<String>>> = HashMap::new();
    for row in rows {
        groups
            .entry(row[column].clone())
            .or_default()
            .push(row.clone());
    }
    groups
}

fn information_gain(rows: &[Vec<String>], feature_column: usize, target_column: usize) -> f64 {
    let base_entropy = entropy(rows, target_column);
    let groups = groups(rows, feature_column);

    let remainder: f64 = groups
        .values()
        .map(|subset| entropy(subset, target_column) * (subset.len() as f64) / (rows.len() as f64))
        .sum();

    base_entropy - remainder
}

fn majority_class(rows: &[Vec<String>], target_column: usize) -> String {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for row in rows {
        *counts.entry(&row[target_column]).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(label, _)| label.to_string())
        .unwrap_or_default()
}

fn id3(
    rows: &[Vec<String>],
    target_column: usize,
    candidate_features: &HashSet<usize>,
) -> DecisionTree {
    let majority_class = majority_class(rows, target_column);

    if rows.is_empty() {
        return DecisionTree::Leaf(majority_class);
    }

    let labels = column_labels(rows, target_column);

    if labels.len() == 1 {
        return DecisionTree::Leaf(labels.iter().next().unwrap().to_string());
    }

    if candidate_features.is_empty() {
        return DecisionTree::Leaf(majority_class);
    }

    let best_feature = *candidate_features
        .iter()
        .max_by(|&&a, &&b| {
            let gain_a = information_gain(rows, a, target_column);
            let gain_b = information_gain(rows, b, target_column);
            gain_a
                .partial_cmp(&gain_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    let groups = groups(rows, best_feature);

    let mut next_features = candidate_features.clone();
    next_features.remove(&best_feature);

    let mut branches: HashMap<String, Box<DecisionTree>> = HashMap::new();
    for (value, subset) in groups {
        branches.insert(value, Box::new(id3(&subset, target_column, &next_features)));
    }

    DecisionTree::Branch {
        column: best_feature,
        default_class: majority_class,
        branches,
    }
}

fn print_tree(tree: &DecisionTree, features: &[String], indent: usize, class: &str) {
    let prefix = "  ".repeat(indent);
    match tree {
        DecisionTree::Leaf(label) => {
            println!("{prefix}-> {class}: {label}");
        }
        DecisionTree::Branch {
            column,
            default_class,
            branches,
        } => {
            let feature_name = &features[*column];
            println!("{prefix}[{feature_name}] (default fallback: {default_class})");
            for (value, subtree) in branches {
                println!("{prefix}  = {value}:");
                print_tree(subtree, features, indent + 2, class);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = parse_args(&args);
    let content =
        std::fs::read_to_string(&config.csv_file_path).expect("Failed to read the CSV file.");
    let (features, rows) = parse_data(&content);
    let target_column = features
        .iter()
        .position(|f| f == &config.target_column)
        .expect("Given target_column doesn't exist in the dataset.");

    let candidate_features: HashSet<usize> = (0..features.len())
        .filter(|&i| i != target_column)
        .collect();

    let tree = id3(&rows, target_column, &candidate_features);

    println!("Decision Tree for target '{}':\n", config.target_column);
    print_tree(&tree, &features, 0, &config.target_column);
}

struct Config {
    csv_file_path: String,
    target_column: String,
}

fn parse_args(args: &[String]) -> Config {
    if args.len() != 3 {
        panic!("Usage: decision-tree-maker <file_path> <target_column>");
    }

    Config {
        csv_file_path: args[1].clone(),
        target_column: args[2].clone(),
    }
}
