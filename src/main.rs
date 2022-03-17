#![allow(dead_code)]
use std::ops::Index;
use std::{cmp, vec};

use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::RngCore;

use std::fs::File;
use std::io::prelude::*;

//Misc Constants to make life easier
const NUM_GEN: i32 = 100;
const MUT_CHANCE: f32 = 0.15;
const CROSS_CHANCE: f32 = 0.85;
const MIN_DIST: u32 = 6;
const N: u32 = 17;
const D: u32 = 6;

//utility functions
fn minimum(a: u32, b: u32, c: u32) -> u32 {
    cmp::min(a, cmp::min(b, c))
}

//Function to find hamming distance between two codewords
//as represented by unsigned 32 bit integers
fn hamming_dist(a: u32, b: u32) -> u32 {
    (a ^ b).count_ones() as u32
}

//Helper function to return a vectore of codewords from a chromosome
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
    //return
    codewords
}

//Helper function to return vector chromosome from vector of codewords
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

//Helper function to check compatibility between codewords when using edit distance metric
fn check_compat(word1: u32, word2: u32, compat_matrix: &Vec<Vec<u32>>) -> bool {
    let indx1 = compat_matrix[0].iter().position(|&x| x == word1).unwrap();
    let indx2 = compat_matrix[0].iter().position(|&y| y == word2).unwrap();
    if compat_matrix[indx1][indx2] == 0 {
        false
    } else {
        true
    }
}

//Edit distance function
//requires use of string slices as input instead of integers so much slower
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

//Helper function used when using edit distance instead of hamming distance
//returns 2D Vector of 1's for compatible words and 0's otherwise
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
            let tmp = edit_distance(&x, &y);
            if tmp >= mindist {
                matrix[i][j] = 1;
            }
        }
    }
    println!("cat");
    matrix
}

//Unused Lexicode function that fitness function was built from
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
                if hamming_dist(candidates[i], *j) < MIN_DIST {
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

//Helper function to return a vector of legal candidate codewords
pub fn create_candidates(n: u32, d: u32) -> Vec<u32> {
    let tmp_codes: Vec<u32> = (0..(2_u32.pow(n))).collect();
    let mut codes: Vec<u32> = vec![0];
    for i in &tmp_codes {
        if hamming_dist(codes[0], *i) >= d {
            codes.push(*i);
        }
    }
    codes
}

//Struct for Individual
//contains:
//chromosome - Vector of unsigned 32 bit integers
//fitness - unsigned 32 bit integer
//chance_mut - 32 bit float
#[derive(Clone)]
pub struct Individual {
    chromosome: Vec<u32>,
    fitness: u32,
    chance_mut: f32,
}
//Methods for Individual Struct
impl Individual {
    //"constructor"
    fn new(chromosome: Vec<u32>, fitness: u32, chance_mut: f32) -> Individual {
        Individual {
            chromosome: chromosome,
            fitness: fitness,
            chance_mut: chance_mut,
        }
    }
    //helper mthods
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

//mutation function
//input includes mutable reference to Individual which is mutated in place
pub fn mutate(rng: &mut dyn RngCore, child: &mut Individual, chance_mut: f32) {
    //check if individual mutates
    if rng.gen_bool(chance_mut as _) {
        //choose and remove element
        let element = rng.gen_range(0..child.chromosome.len());
        child.chromosome.remove(element);

        //create candidate words
        let candidates = create_candidates(N, D);
        let mut child_codes = words_from_chrom(&child.chromosome, &candidates);
        let tmp_child_codes = child_codes.clone();
        let mut done: bool = false;

        //logic to find a compatible replacement word and insert it into the chromosome of the child Individual
        for i in &candidates {
            if done {
                break;
            }
            for j in &tmp_child_codes {
                if done {
                    break;
                }
                if *i == *j {
                    break;
                } else if hamming_dist(*i, *j) < MIN_DIST {
                    break;
                } else {
                    match child_codes.binary_search(&i) {
                        Ok(_pos) => {}
                        Err(pos) => child_codes.insert(pos, *i),
                    }
                    child.chromosome = chrom_from_words(&child_codes, &candidates);
                    done = true;
                }
            }
        }
    }
}

//Population struct
//contains vector of individuals
#[derive(Clone)]
pub struct Population {
    pop: Vec<Individual>,
}

//methods for Population struct
impl Population {
    //generates a population of individuals with legal chromosomes
    fn gen_pop(popsize: u32, n: u32, chrom_size: u32, chance_mut: f32) -> Vec<Individual> {
        let mut rng = rand::thread_rng();
        let candidates: Vec<u32> = create_candidates(n, MIN_DIST);
        let mut pop: Vec<Individual> = vec![];
        let _idx: usize = 0;

        //filling population
        while pop.len() < (popsize as usize) {
            let mut tmp: Vec<u32> = vec![0];
            let _tmp_idx: usize = 0;
            while tmp.len() < (chrom_size as usize) {
                let indx = rng.gen_range(1..candidates.len());
                for i in &tmp {
                    if hamming_dist(*i, candidates[indx]) < MIN_DIST {
                        break;
                    }
                }
                tmp.push(candidates[indx]);
            }
            //init individual to add to population
            let tmp_ind = Individual::new(tmp, 0_u32, chance_mut);
            pop.push(tmp_ind);
        }
        //return population
        pop
    }

    //Standard Tournament Selection implementation
    // # Arguments
    // * `rng` - mutable reference to random number generator
    // * `population` - reference to vector containing Individuals
    // * `tourn_size` - tournament size
    // * Returns an Individual
    pub fn tournament_select(
        rng: &mut dyn RngCore,
        population: &Vec<Individual>,
        tourn_size: usize,
    ) -> Individual {
        let mut tourn: Vec<Individual> = vec![];
        while tourn.len() < tourn_size {
            let indiv = population.choose(rng).expect("got an empty population");
            tourn.push(indiv.clone());
        }
        let child = tourn.iter().max_by_key(|f| f.fitness).unwrap().clone();
        child
    }

    //Roulette Wheel Selection implementation
    pub fn roulette_select(rng: &mut dyn RngCore, population: &Vec<Individual>) -> Individual {
        let _tmp = population.clone();

        //Find total population fitness
        let total_fitness: f32 = population
            .iter()
            .map(|individual| individual.fitness as f32)
            .sum();

        //Loop through population and select winner
        loop {
            let indiv = population.choose(rng).expect("got an empty population");

            let indiv_share = indiv.fitness as f32 / total_fitness;

            if rng.gen_bool(indiv_share as f64) {
                return indiv.clone();
            }
        }
    }

    //Single point crossover
    //returns single individual
    pub fn sp_crossover(
        rng: &mut dyn RngCore,
        parent_1: &Individual,
        parent_2: &Individual,
        cross_chance: f32,
    ) -> Individual {
        //check if crossover occurs
        if rng.gen_bool(cross_chance as _) {
            let candidates = create_candidates(N, D);
            //sanity checks
            if parent_1.len() != parent_2.len() {
                if parent_1.len() > parent_2.len() {
                    let new = Individual::new(parent_1.chromosome.clone(), 0, MUT_CHANCE);
                    return new;
                } else {
                    let new = Individual::new(parent_2.chromosome.clone(), 0, MUT_CHANCE);
                    return new;
                }
            }
            //pick crossover point
            let crossoverpoint = Uniform::new(0, parent_1.chromosome.len()).sample(rng);
            let mut child: Vec<u32> = parent_1.chromosome.clone();
            //remove data after crossover point
            let _dispose: Vec<u32> = child.drain(crossoverpoint..).collect();
            let mut p2_dna = parent_2.chromosome.clone();
            //remove data before crossover point
            p2_dna = p2_dna.drain(crossoverpoint..).collect();
            let mut child_words = words_from_chrom(&child, &candidates);
            let p2_dna_words = words_from_chrom(&p2_dna, &candidates);

            //iterate through vector of words from 2nd parent to find ones to insert into child
            for i in &p2_dna_words {
                let mut flag: bool = true;
                for j in &child_words {
                    if hamming_dist(*i, *j) < MIN_DIST {
                        flag = false;
                        break;
                    }
                }
                if flag {
                    child_words.push(*i);
                }
            }
            child_words.sort();
            let child = chrom_from_words(&child_words, &candidates);
            let new = Individual::new(child, 0_u32, MUT_CHANCE);
            new
        } else {
            let new = Individual::new(parent_1.chromosome.clone(), 0, MUT_CHANCE);
            new
        }
    }
}

//greedy lexicode algorithm for fitness
//returns a tuple of the resultant codewords and the length of the codeword vector
fn fitness(lexicodes: &mut Vec<u32>, cand: &Vec<u32>) -> (Vec<u32>, u32) {
    let clength = lexicodes.len();
    let mut candidates = cand.clone();
    //find max value of the seed codes
    let maxvalue = lexicodes.iter().max().unwrap();
    //find place in candidate codeword vector to begin iterating from for new codewords
    let indx1 = candidates.iter().position(|&x| x == *maxvalue).unwrap();
    candidates = candidates.drain((indx1 + 1)..).collect();
    //iterate through candidate codewords to find words compatible to all others in the set
    for i in 0..candidates.len() {
        if clength >= 1 {
            let mut flag: bool = true;
            for j in lexicodes.iter() {
                if hamming_dist(candidates[i], *j) < MIN_DIST {
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

//Co-ordination function for the mating events
//parents are selected, followed by crossover and finally mutation
//process repeats until length of population
pub fn evolve(
    rng: &mut dyn RngCore,
    population: &Vec<Individual>,
    chance_cross: f32,
    chance_mut: f32,
    tourn_size: usize,
) -> Vec<Individual> {
    assert!(!population.is_empty());
    let mut new_pop: Vec<Individual> = vec![];

    //elitism implementation
    let elite: &Individual = population.iter().max_by_key(|f| f.fitness).unwrap();
    let test = elite.clone();
    new_pop.push(test);

    for _ in 1..population.len() {
        let parent_1 = Population::tournament_select(rng, population, tourn_size);
        let parent_2 = Population::tournament_select(rng, population, tourn_size);
        // let parent_1 = Population::roulette_select(rng, population);
        // let parent_2 = Population::roulette_select(rng, population);
        let mut child = Population::sp_crossover(rng, &parent_1, &parent_2, chance_cross);
        mutate(rng, &mut child, chance_mut);
        new_pop.push(child)
    }
    new_pop
}

//Helper function to find and set the population values for each individual of population
pub fn get_fitness(populat: &mut Population) {
    let candidates = create_candidates(N, D);
    for mut i in &mut populat.pop {
        let mut words = words_from_chrom(&i.chromosome, &candidates);
        if words.len() < 1 {
            i.fitness = 0;
        } else {
            let (_codes, fit) = fitness(&mut words, &candidates);
            i.fitness = fit;
        }
    }
}

//Entry point to Program
fn main() {
    //Random number generator
    let mut rng = rand::thread_rng();

    //GA initialization
    let pop1 = Population::gen_pop(500, N, 330, MUT_CHANCE);
    let candidates = create_candidates(N, D);
    let mut population = Population { pop: pop1 };

    //establish initial population fitness
    get_fitness(&mut population);

    //GA loop
    for i in 0..NUM_GEN {
        population.pop = evolve(&mut rng, &population.pop, CROSS_CHANCE, MUT_CHANCE, 3);
        get_fitness(&mut population);
        println!("Gen: {}", i);
    }

    // I/O
    let mut file = File::create("results.txt").expect("create failed");

    //Output control fitness based on zero vector
    let mut control_vec: Vec<u32> = vec![0];
    let (control_codes, control_fit) = fitness(&mut control_vec, &candidates);
    println!("Starting with 0 vector, fit was: {}", control_fit);
    let result = format!("Starting with 0 vector, fit was: {}", control_fit);
    writeln!(file, "{}", result).expect("write failed");

    //More I/O
    let mut best_fit = 0;
    for i in population.pop {
        println!("number words: {}  ", i.fitness);
        if i.fitness == 0 {
            continue;
        }
        let mut seed_words = words_from_chrom(&i.chromosome, &candidates);
        let (words, fit) = fitness(&mut seed_words, &candidates);
        if fit > best_fit {
            best_fit = fit;
        }
        let result = format!("Number of words: {}", fit);
        writeln!(file, "{}", result).expect("write failed");
        for j in &words {
            println!(" words: {:017b}", *j);
        }
    }
    let besty = format!("Best fit was: {}", best_fit);
    writeln!(file, "{}", besty).expect("msg");

    println!("Hello, world!");
}
