use std::cmp;
use std::iter::FromIterator;
use std::ops::Index;

use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// fn UniformCrossover(
//     rng: &mut dyn RngCore,
//     parent_1: &Chromosome,
//     parent_2: &Chromosome,
//     compat_matrix: &Vec<Vec<u32>>,
// ) -> Chromosome {
//     assert_eq!(parent_1.len(), parent_2.len());
//     if rng.gen_bool(self.chance_cross as _) {
//         let mut cloned_matrix = compat_matrix.clone();
//         let mut child: Vec<u32> = Vec::new();
//         let num_genes = parent_1.len();
//         let mut cand_codes_vec: Vec<u32> = vec![];
//         let mut gene_idx = 0;
//         // loop until chromosome is full
//         while gene_idx < num_genes {
//             let gene = if rng.gen_bool(0.5) {
//                 parent_1[gene_idx]
//             } else {
//                 parent_2[gene_idx]
//             };
//             if child.len() == 0 {
//                 child.push(gene);
//                 cand_codes_vec = cloned_matrix[0].drain((gene + 1) as usize..).collect();
//             } else {
//                 let mut flag: bool = true;

//                 for j in &child {
//                     let new_code = cand_codes_vec[*j as usize];
//                     let indx = compat_matrix[0]
//                         .iter()
//                         .position(|&x| x == new_code)
//                         .unwrap();
//                     let child_indx = compat_matrix[0]
//                         .iter()
//                         .position(|&y| y == cand_codes_vec[gene as usize])
//                         .unwrap();
//                     if compat_matrix[indx][child_indx] == 0 {
//                         flag = false;
//                         continue;
//                     }
//                 }
//                 if flag == true {
//                     child.push(gene);
//                     cand_codes_vec = cand_codes_vec.drain((gene + 1) as usize..).collect();
//                     gene_idx += 1;
//                 }
//             }
//         }
//         child.into_iter().collect()
//     } else {
//         let child: Vec<u32> = parent_1.genes.clone();
//         child.into_iter().collect()
//     }
// }

// pub trait MutationMethod {
//     fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome, compat_matrix: &Vec<Vec<u32>>);
// }

// #[derive(Clone, Debug)]
// pub struct CodeMutation {
//     chance_mut: f32,
// }

// impl CodeMutation {
//     pub fn new(chance_mut: f32) -> Self {
//         assert!(chance_mut >= 0.0 && chance_mut <= 1.0);
//         Self { chance_mut }
//     }
// }

// impl MutationMethod for CodeMutation {
//     fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome, compat_matrix: &Vec<Vec<u32>>) {
//         if rng.gen_bool(self.chance_mut as _) {
//             let element = rng.gen_range(0..child.genes.len());
//             child.genes.remove(element);
//             let mut cloned_matrix = compat_matrix[0].clone();
//             cloned_matrix.remove(0);
//             let mut child_codes = words_from_chrom(&child.genes, &cloned_matrix);
//             let tmp_child_codes = child_codes.clone();
//             let mut done: bool = false;
//             for i in &cloned_matrix {
//                 if done {
//                     break;
//                 }
//                 for j in &tmp_child_codes {
//                     if done {
//                         break;
//                     }
//                     if *i == *j {
//                         break;
//                     } else if !check_compat(*i, *j, compat_matrix) {
//                         break;
//                     } else {
//                         match child_codes.binary_search(&i) {
//                             Ok(_pos) => {}
//                             Err(pos) => child_codes.insert(pos, *i),
//                         }
//                         child.genes = chrom_from_words(&child_codes, &cloned_matrix);
//                         done = true;
//                     }
//                 }
//             }
//         }
//     }
// }

// pub trait SelectionMethod {
//     fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
//     where
//         I: Individual;
// }

// #[derive(Clone, Debug, Default)]
// pub struct RouletteWheelSelection;

// impl SelectionMethod for RouletteWheelSelection {
// fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
// where
//     I: Individual,
// {
//     population
//         .choose_weighted(rng, |individual| individual.fitness())
//         .expect("got an empty population")
// }
// }

//utility functions
fn minimum(a: u32, b: u32, c: u32) -> u32 {
    cmp::min(a, cmp::min(b, c))
}

fn words_from_chrom(chrome: &Vec<u32>, candidates: &Vec<u32>) -> Vec<u32> {
    let mut working_candidates = candidates.clone();
    let mut codewords: Vec<u32> = vec![];
    for i in 0..chrome.len() {
        if working_candidates.len() == 0 {
            break;
        } else if working_candidates.len() == 1 {
            codewords.push(working_candidates[0]);
            break;
        } else if chrome[i] as usize > working_candidates.len() - 1 {
            codewords.push(working_candidates.pop().unwrap());
            working_candidates.clear();
            break;
        } else {
            codewords.push(working_candidates[chrome[i] as usize]);
            if working_candidates.len() > 1 {
                working_candidates = working_candidates
                    .drain((chrome[i] + 1) as usize..)
                    .collect();
            }
        }
    }
    codewords
}

fn chrom_from_words(codes: &Vec<u32>, candidates: &Vec<u32>) -> Vec<u32> {
    let mut result: Vec<u32> = vec![];
    let mut cand_clone = candidates.clone();
    for i in codes {
        let indx = cand_clone.iter().position(|&x| x == *i).unwrap();
        result.push(indx as u32);
        cand_clone = cand_clone.drain((indx + 1)..).collect();
    }
    result
}

fn check_compat(word1: u32, word2: u32, compat_matrix: &Vec<Vec<u32>>) -> bool {
    let indx1 = compat_matrix[0].iter().position(|&x| x == word1).unwrap();
    let indx2 = compat_matrix[0].iter().position(|&y| y == word2).unwrap();
    if compat_matrix[indx1][indx2] == 0 {
        false
    } else {
        true
    }
}

fn edit_distance(s1: &str, s2: &str) -> u32 {
    // get length of unicode chars
    let len_s = s1.chars().count();
    let len_t = s2.chars().count();

    // initialize the matrix
    let mut mat: Vec<Vec<u32>> = vec![vec![0; len_t + 1]; len_s + 1];
    for i in 1..(len_s + 1) {
        mat[i][0] = i as u32;
    }
    for i in 1..(len_t + 1) {
        mat[0][i] = i as u32;
    }

    // apply edit operations
    for (i, s1_char) in s1.chars().enumerate() {
        for (j, s2_char) in s2.chars().enumerate() {
            let substitution = if s1_char == s2_char { 0 } else { 1 };
            mat[i + 1][j + 1] = minimum(
                mat[i][j + 1] + 1,        // deletion
                mat[i + 1][j] + 1,        // insertion
                mat[i][j] + substitution, // substitution
            );
        }
    }

    return mat[len_s][len_t];
}

fn compat_matrix(codes: &Vec<u32>, mindist: u32) -> Vec<Vec<u32>> {
    let precomp = codes.clone();
    let mut precomp2: Vec<u32> = Vec::new();
    for i in &precomp {
        let x = format!("{:016b}", precomp[0]);
        let y = format!("{:016b}", *i);
        let tmp = edit_distance(&x, &y);
        if tmp >= mindist {
            precomp2.push(*i);
        }
    }
    precomp2.insert(0, 0);
    let mut matrix = vec![vec![0u32; precomp2.len() + 1]; precomp2.len() + 1];
    matrix[0] = precomp2.clone();
    matrix[0].insert(0, 0);
    for i in 2..=precomp2.len() {
        matrix[i][0] = precomp2[i - 1];
    }
    for i in 1..matrix[0].len() {
        for j in 1..matrix.len() {
            let x = format!("{:016b}", matrix[i][0]);
            let y = format!("{:016b}", matrix[0][j]);
            // let a = matrix[i][0];
            // let b = matrix[0][j];
            let tmp = edit_distance(&x, &y);
            if tmp >= mindist {
                matrix[i][j] = 1;
            }
        }
    }
    println!("cat");
    matrix
}

fn lexicode(lexicodes: &mut Vec<u32>, compat: &Vec<Vec<u32>>) -> Vec<u32> {
    let clength = lexicodes.len();
    let mut candidates: Vec<u32> = compat[0].clone();
    let maxvalue = candidates.iter().max().unwrap();
    let indx1 = candidates.iter().position(|&x| x == *maxvalue).unwrap();
    candidates = candidates.drain((indx1 + 1)..).collect();
    for i in 0..candidates.len() {
        if clength >= 1 {
            let mut flag: bool = true;
            for j in lexicodes.iter() {
                if !check_compat(candidates[i], *j, compat) {
                    flag = false;
                    continue;
                }
            }
            if flag == true {
                lexicodes.push(candidates[i]);
            }
        }
    }
    return lexicodes.to_vec();
}

pub fn create_candidates(n: u32) -> Vec<u32> {
    let codes: Vec<u32> = (0..(2_u32.pow(n))).collect();
    codes
}

pub struct Individual {
    chromosome: Vec<u32>,
    fitness: u32,
}

impl Individual {
    fn new(chromosome: Vec<u32>, fitness: u32) -> Individual {
        Individual {
            chromosome: chromosome,
            fitness: fitness,
        }
    }
    //greedy lexicode algorithm for fitness
    fn fitness(lexicodes: &mut Vec<u32>, compat: &Vec<Vec<u32>>) -> (Vec<u32>, u32) {
        let clength = lexicodes.len();
        let mut candidates: Vec<u32> = compat[0].clone();
        let maxvalue = candidates.iter().max().unwrap();
        let indx1 = candidates.iter().position(|&x| x == *maxvalue).unwrap();
        candidates = candidates.drain((indx1 + 1)..).collect();
        for i in 0..candidates.len() {
            if clength >= 1 {
                let mut flag: bool = true;
                for j in lexicodes.iter() {
                    if !check_compat(candidates[i], *j, compat) {
                        flag = false;
                        continue;
                    }
                }
                if flag == true {
                    lexicodes.push(candidates[i]);
                }
            }
        }
        let fit: u32 = lexicodes.len() as u32;
        return (lexicodes.to_vec(), fit);
    }
}

pub struct Population {
    pop: Vec<Individual>,
}

impl Population {
    fn gen_pop(popsize: u32, n: u32, d: u32, chrom_size: u32) -> (Vec<Individual>, Vec<Vec<u32>>) {
        let mut rng = rand::thread_rng();
        let cand: Vec<u32> = create_candidates(n);
        let compat = compat_matrix(&cand, d);
        let mut candidates = compat[0].clone();
        candidates = candidates.drain((1)..).collect();
        let mut pop: Vec<Individual> = vec![];
        let idx: usize = 0;

        //filling population
        while pop.len() < (popsize as usize) {
            let mut tmp: Vec<u32> = vec![0];
            // pop.push(tmp);
            let tmp_idx: usize = 0;
            while tmp.len() < (chrom_size as usize) {
                let indx = rng.gen_range(1..candidates.len());
                for i in &tmp {
                    if check_compat(*i, candidates[indx], &compat) == false {
                        break;
                    }
                }
                tmp.push(candidates[indx]);
            }
            //init individual to add to population
            let tmp_ind = Individual::new(tmp, 0_u32);
            pop.push(tmp_ind);
        }
        (pop, compat)
    }

    pub fn roulette_select(rng: &mut dyn RngCore, population: &Vec<Individual>) -> Individual {
        let tmp = population.clone();
        let total_fitness: f32 = population
            .iter()
            .map(|individual| individual.fitness as f32)
            .sum();
        loop {
            let indiv = population.choose(rng).expect("got an empty population");

            let indiv_share = indiv.fitness as f32 / total_fitness;

            if rng.gen_bool(indiv_share as f64) {
                return *indiv;
            }
        }
    }
}

pub fn evolve(
    rng: &mut dyn RngCore,
    population: &[I],
    compat_matrix: &Vec<Vec<u32>>,
    // chance_cross: f64,
) -> Vec<Individual> {
    assert!(!population.is_empty());
    (0..population.len())
        .map(|_| {
            let parent_1 = Population::roulette_select(rng, population);
            let parent_2 = Population::roulette_select(rng, population);
            let mut child = self
                .crossover_meth
                .crossover(rng, parent_1, parent_2, compat_matrix);

            self.mutation_meth.mutate(rng, &mut child, compat_matrix);
        })
        .collect()
}

fn main() {
    let codes: Vec<u32> = create_candidates(6);
    // let c_matrix = compat_matrix(&codes, 3);
    let mut rng = ChaCha8Rng::from_seed(Default::default());
    let init: Vec<Individual> = vec![];
    let mut test = Population { pop: init };
    let (pop1, compat_matrix) = Population::gen_pop(5, 6, 3, 7);
    let population = Population { pop: pop1 };

    //GA loop

    // let genetic_algo = GeneticAlgorithm::new(
    //     RouletteWheelSelection::default(),
    //     UniformCrossover::new(0.85),
    //     CodeMutation::new(0.15),
    // );

    // let bit_test = bitmask_test(&codes, 3);
    // for i in &codes {
    //     println!("{:08b}", i);
    // }
    // let test = codes[11] ^ codes[14];
    // let test2 = compat_matrix(&codes, 3);
    let mut codeinit: Vec<u32> = vec![0];
    // let codie_edit = lexicode(&mut codeinit, &test2);
    // let codie_edit = lexicode3(5, &codes, &mut codeinit);
    // for i in &codie_edit {
    //     println!("{:012b}", i);
    //     println!("int: {}", i);
    // }
    // let codies = lexicode2(6, &codes, &mut codeinit);
    // for i in &codies {
    //     println!("{:017b}", i);
    //     println!("int: {}", i);
    // }
    // println!("{}", codie_edit.len());
    println!("Hello, world!");
    // println!("{:08b}", test);
    // let x = format!("{:08b}", codes[5]);
    // let x_vec: Vec<char> = x.chars().collect();
    // let y = format!("{:08b}", codes[2]);
    // let y_vec: Vec<char> = y.chars().collect();

    // let testy = levenshtein_d1stance(&x, &y);
    // let s = format!("{:08b}", x);
    // println!("x is: {:?} or {}", x_vec, codes[5]);
    // println!("y is: {:?} or {}", y_vec, codes[2]);
    // println!("The levenshtein dist is: {}", testy);
    // println!("Hamming distance is: {}", hamming_dist(codes[5], codes[2]));
    // println!("Edit distance: {}", edit_distance(&x, &y));
    // let inty = s.parse::<u32>().unwrap();

    // println!("{}", inty);
}
