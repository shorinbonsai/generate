#![allow(dead_code)]
// use rand_core::RngCore;
use std::cmp;
use std::iter::FromIterator;
use std::ops::Index;

use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;
// use rand_chacha::ChaCha8Rng;

const NUM_GEN: i32 = 100;
const MUT_CHANCE: f32 = 0.15;
const CROSS_CHANCE: f32 = 0.85;

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
#[derive(Clone)]
pub struct Individual {
    chromosome: Vec<u32>,
    fitness: u32,
    chance_mut: f32,
}

impl Individual {
    fn new(chromosome: Vec<u32>, fitness: u32, chance_mut: f32) -> Individual {
        Individual {
            chromosome: chromosome,
            fitness: fitness,
            chance_mut: chance_mut,
        }
    }

    pub fn len(&self) -> usize {
        self.chromosome.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &u32> {
        self.chromosome.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut u32> {
        self.chromosome.iter_mut()
    }
}

pub fn mutate(
    rng: &mut dyn RngCore,
    child: &mut Individual,
    compat_matrix: &Vec<Vec<u32>>,
    chance_mut: f32,
) {
    if rng.gen_bool(chance_mut as _) {
        let element = rng.gen_range(0..child.chromosome.len());
        child.chromosome.remove(element);
        let mut cloned_matrix = compat_matrix[0].clone();
        cloned_matrix.remove(0);
        let mut child_codes = words_from_chrom(&child.chromosome, &cloned_matrix);
        let tmp_child_codes = child_codes.clone();
        let mut done: bool = false;
        for i in &cloned_matrix {
            if done {
                break;
            }
            for j in &tmp_child_codes {
                if done {
                    break;
                }
                if *i == *j {
                    break;
                } else if !check_compat(*i, *j, compat_matrix) {
                    break;
                } else {
                    match child_codes.binary_search(&i) {
                        Ok(_pos) => {}
                        Err(pos) => child_codes.insert(pos, *i),
                    }
                    child.chromosome = chrom_from_words(&child_codes, &cloned_matrix);
                    done = true;
                }
            }
        }
    }
}

impl Index<usize> for Individual {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.chromosome[index]
    }
}
#[derive(Clone)]
pub struct Population {
    pop: Vec<Individual>,
}

impl Population {
    fn gen_pop(
        popsize: u32,
        n: u32,
        d: u32,
        chrom_size: u32,
        chance_mut: f32,
    ) -> (Vec<Individual>, Vec<Vec<u32>>) {
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
            let tmp_ind = Individual::new(tmp, 0_u32, chance_mut);
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
                return indiv.clone();
            }
        }
    }

    pub fn UniformCrossover(
        rng: &mut dyn RngCore,
        parent_1: &Individual,
        parent_2: &Individual,
        compat_matrix: &Vec<Vec<u32>>,
        cross_chance: f32,
    ) -> Individual {
        if rng.gen_bool(cross_chance as _) {
            let crossoverpoint = Uniform::new(1, parent_1.chromosome.len()).sample(rng);
            let mut candidates = compat_matrix[0].clone();
            candidates = candidates.drain(1..).collect();
            let mut child: Vec<u32> = parent_1.chromosome.clone();
            let dispose: Vec<u32> = child.drain(crossoverpoint..).collect();
            let mut p2_dna = parent_2.chromosome.clone();
            p2_dna = p2_dna.drain(crossoverpoint..).collect();
            let mut child_words = words_from_chrom(&child, &candidates);
            let mut p2_dna_words = words_from_chrom(&p2_dna, &candidates);

            for i in &p2_dna_words {
                let mut flag: bool = true;
                for j in &child_words {
                    if !check_compat(*i, *j, &compat_matrix) {
                        flag = false;
                    }
                }
            }
            child.extend_from_slice(&p2_dna);
            let new = Individual::new(child, 0_u32, MUT_CHANCE);
            new
        } else {
            let new = Individual::new(parent_1.chromosome.clone(), 0, MUT_CHANCE);
            new
        }
        // assert_eq!(parent_1.chromosome.len(), parent_2.chromosome.len());
        //     if parent_1.chromosome.len() != parent_2.chromosome.len() {
        //         println!("Diff length chromosome...");
        //     }
        //     if rng.gen_bool(cross_chance as _) {
        //         let mut cloned_matrix = compat_matrix.clone();
        //         let mut child: Vec<u32> = Vec::new();
        //         let num_genes = parent_1.chromosome.len();
        //         let mut cand_codes_vec: Vec<u32> = vec![];
        //         let mut gene_idx = 1;
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
        //         let mut child_individual = Individual::new(child, 0, cross_chance);
        //         child_individual
        //     } else {
        //         let child: Vec<u32> = parent_1.chromosome.clone();
        //         let mut child_individual = Individual::new(child, 0, cross_chance);
        //         child_individual
        //     }
    }
}

//greedy lexicode algorithm for fitness
fn fitness(lexicodes: &mut Vec<u32>, compat: &Vec<Vec<u32>>) -> (Vec<u32>, u32) {
    let clength = lexicodes.len();
    let mut candidates: Vec<u32> = compat[0].clone();
    let maxvalue = lexicodes.iter().max().unwrap();
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

pub fn evolve(
    rng: &mut dyn RngCore,
    population: &Vec<Individual>,
    compat_matrix: &Vec<Vec<u32>>,
    chance_cross: f32,
    chance_mut: f32,
) -> Vec<Individual> {
    assert!(!population.is_empty());
    let mut new_pop: Vec<Individual> = vec![];
    // let elite = vec![];
    for _ in 1..population.len() {
        let parent_1 = Population::roulette_select(rng, population);
        let parent_2 = Population::roulette_select(rng, population);
        let mut child =
            Population::UniformCrossover(rng, &parent_1, &parent_2, compat_matrix, chance_cross);
        mutate(rng, &mut child, compat_matrix, chance_mut);
        new_pop.push(child)
    }
    new_pop
}

pub fn get_fitness(populat: &mut Population, compat_matrix: &Vec<Vec<u32>>) {
    let mut candidates = compat_matrix[0].clone();
    let _discard = candidates.remove(0);
    for mut i in &mut populat.pop {
        let mut words = words_from_chrom(&i.chromosome, &candidates);
        let (_codes, fit) = fitness(&mut words, &compat_matrix);
        i.fitness = fit;
    }
}

fn main() {
    let codes: Vec<u32> = create_candidates(6);
    // let c_matrix = compat_matrix(&codes, 3);
    let mut rng = rand::thread_rng();
    let init: Vec<Individual> = vec![];
    let mut test = Population { pop: init };

    //GA init
    let (pop1, compat_matrix) = Population::gen_pop(50, 6, 3, 7, MUT_CHANCE);
    let mut candidates = compat_matrix[0].clone();
    candidates = candidates.drain(1..).collect();
    let mut population = Population { pop: pop1 };
    get_fitness(&mut population, &compat_matrix);
    //GA loop
    for i in 0..NUM_GEN {
        evolve(
            &mut rng,
            &population.pop,
            &compat_matrix,
            CROSS_CHANCE,
            MUT_CHANCE,
        );
        get_fitness(&mut population, &compat_matrix);
        println!("Gen: {}", i);
    }
    for i in population.pop {
        println!("number words: {}  ", i.fitness);
        let mut seed_words = words_from_chrom(&i.chromosome, &candidates);
        let (mut words, fit) = fitness(&mut seed_words, &compat_matrix);
        for j in &words {
            println!(" words: {:06b}", *j);
        }
    }

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
}
