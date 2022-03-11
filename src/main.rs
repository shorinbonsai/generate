use std::cmp;
use std::iter::FromIterator;
use std::ops::Index;

use rand::seq::SliceRandom;
use rand::Rng;
use rand::RngCore;

pub struct GeneticAlgorithm<S, C> {
    selection_meth: S,
    crossover_meth: C,
}

impl<S, C> GeneticAlgorithm<S, C>
where
    S: SelectionMethod,
    C: CrossoverMethod,
{
    pub fn new(selection_meth: S, crossover_meth: C) -> Self {
        Self {
            selection_meth,
            crossover_meth,
        }
    }

    pub fn evolve<I>(
        &self,
        rng: &mut dyn RngCore,
        population: &[I],
        compat_matrix: &Vec<Vec<u32>>,
        chance_cross: f64,
    ) -> Vec<I>
    where
        I: Individual,
    {
        assert!(!population.is_empty());
        (0..population.len())
            .map(|_| {
                let parent_1 = self.selection_meth.select(rng, population).chromosome();
                let parent_2 = self.selection_meth.select(rng, population).chromosome();
                if rng.gen_bool(chance_cross) {
                    let mut child =
                        self.crossover_meth
                            .crossover(rng, parent_1, parent_2, compat_matrix);
                } else {
                    let mut child = parent_1;
                }

                // TODO mutation
                todo!()
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct Chromosome {
    genes: Vec<u32>,
}
impl Chromosome {
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &u32> {
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut u32> {
        self.genes.iter_mut()
    }
}

impl Index<usize> for Chromosome {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl FromIterator<u32> for Chromosome {
    fn from_iter<T: IntoIterator<Item = u32>>(iter: T) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for Chromosome {
    type Item = u32;
    type IntoIter = std::vec::IntoIter<u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

pub trait Individual {
    fn chromosome(&self) -> &Chromosome;
    fn fitness(&self) -> f32;
}

pub trait CrossoverMethod {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_1: &Chromosome,
        parent_2: &Chromosome,
        compat_matrix: &Vec<Vec<u32>>,
    ) -> Chromosome;
}

#[derive(Clone, Debug)]
pub struct UniformCrossover;

impl UniformCrossover {
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverMethod for UniformCrossover {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_1: &Chromosome,
        parent_2: &Chromosome,
        compat_matrix: &Vec<Vec<u32>>,
    ) -> Chromosome {
        assert_eq!(parent_1.len(), parent_2.len());
        let mut cloned_matrix = compat_matrix.clone();
        let mut child: Vec<u32> = Vec::new();
        let num_genes = parent_1.len();
        let mut cand_codes_vec: Vec<u32> = vec![];
        let mut gene_idx = 0;
        // loop until chromosome is full
        while gene_idx < num_genes {
            let gene = if rng.gen_bool(0.5) {
                parent_1[gene_idx]
            } else {
                parent_2[gene_idx]
            };
            if child.len() == 0 {
                child.push(gene);
                cand_codes_vec = cloned_matrix[0].drain((gene + 1) as usize..).collect();
            } else {
                let mut flag: bool = true;

                for j in &child {
                    let new_code = cand_codes_vec[*j as usize];
                    let indx = compat_matrix[0]
                        .iter()
                        .position(|&x| x == new_code)
                        .unwrap();
                    let child_indx = compat_matrix[0]
                        .iter()
                        .position(|&y| y == cand_codes_vec[gene as usize])
                        .unwrap();
                    if compat_matrix[indx][child_indx] == 0 {
                        flag = false;
                        continue;
                    }
                }
                if flag == true {
                    child.push(gene);
                    cand_codes_vec = cand_codes_vec.drain((gene + 1) as usize..).collect();
                    gene_idx += 1;
                }
            }
        }

        child.into_iter().collect()
    }
}

pub trait MutationMethod {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome, compat_matrix: &Vec<Vec<u32>>);
}

#[derive(Clone, Debug)]
pub struct CodeMutation {
    chance_mut: f32,
}

impl CodeMutation {
    pub fn new(chance_mut: f32) -> Self {
        assert!(chance_mut >= 0.0 && chance_mut <= 1.0);
        Self { chance_mut }
    }
}

impl MutationMethod for CodeMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome, compat_matrix: &Vec<Vec<u32>>) {
        let element = rng.gen_range(0..child.genes.len());
        child.genes.remove(element);
        let mut tmp_code = vec![];
        let mut cloned_matrix = compat_matrix[0].clone();
        cloned_matrix.remove(0);
        tmp_code.push(compat_matrix[0][child.genes[0] as usize]);
        // for i in &compat_matrix[0] {
        //     if *i < compat_matrix[0][j]
        // }
    }
}

pub trait SelectionMethod {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual;
}

#[derive(Clone, Debug, Default)]
pub struct RouletteWheelSelection;

impl SelectionMethod for RouletteWheelSelection {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual,
    {
        population
            .choose_weighted(rng, |individual| individual.fitness())
            .expect("got an empty population")
    }
}

//utility functions
fn minimum(a: u32, b: u32, c: u32) -> u32 {
    cmp::min(a, cmp::min(b, c))
}

fn words_from_chrom(chrome: &Vec<u32>, candidates: &Vec<u32>) -> Vec<u32> {
    let mut working_candidates = candidates.clone();
    let mut codewords: Vec<u32> = vec![];
    for i in 0..chrome.len() {
        codewords.push(working_candidates[chrome[i] as usize]);
        if working_candidates.len() > 1 {
            working_candidates = working_candidates.drain((chrome[i] + 1) as usize..).collect();
        }
        
    }
    todo!()
}

fn hamming_dist(a: u32, b: u32) -> u32 {
    (a ^ b).count_ones() as u32
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

fn bitmask_test(codes: &Vec<u32>, mindist: u32) {
    let mut codeinit: Vec<u32> = vec![0];
    let precomp = codes.clone();
    let mut precomp2: Vec<u32> = Vec::new();
    for i in &precomp {
        let x = format!("{:06b}", codeinit[0]);
        let y = format!("{:06b}", *i);
        let tmp = edit_distance(&x, &y);
        if tmp >= mindist {
            codeinit.push(*i);
        }
    }
    println!("{:?}", codeinit);
    let x = format!("{:06b}", 3);
    let y = format!("{:06b}", 3);
    let tmp = edit_distance(&x, &y);
    println!("{}", tmp);
    println!(" 1 bits for 6 and 7: {}", (3_u32 & 3_u32).count_ones());
    let mut new_test = vec![0];
    for i in &codeinit {
        for j in &codeinit {
            let a = format!("{:06b}", &i);
            let b = format!("{:06b}", &j);
            if edit_distance(&a, &b) >= mindist {
                new_test.push(*j);
            }
        }
        let x = format!("{:06b}", codeinit[0]);
        let y = format!("{:06b}", *i);
    }
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
    for i in 0..compat[0].len() {
        if clength >= 1 {
            let mut flag: bool = true;
            for j in lexicodes.iter() {
                let indx = compat[0].iter().position(|&x| x == *j).unwrap();
                if compat[indx][i] == 0 {
                    flag = false;
                    continue;
                }
            }
            if flag == true {
                lexicodes.push(compat[0][i]);
                println!("adding: {:016b}", compat[0][i]);
            }
        }
    }
    return lexicodes.to_vec();
}

fn lexicode3(min_dist: u32, codewords: &Vec<u32>, lexicodes: &mut Vec<u32>) -> Vec<u32> {
    let clength = lexicodes.len();
    for i in 0..codewords.len() {
        if clength >= 1 {
            // let mut tmpvec: Vec<u32> = Vec::new();
            let mut flag: bool = true;
            for j in lexicodes.iter() {
                if codewords[i] <= *j {
                    flag = false;
                    continue;
                }
                // let indx = codewords.iter().position(|&x| x == *j).unwrap();
                let x = format!("{:012b}", codewords[i]);
                let y = format!("{:012b}", *j);
                if edit_distance(&x, &y) < min_dist {
                    flag = false;
                    continue;
                }
            }
            if flag == true {
                lexicodes.push(codewords[i]);
                println!("adding: {:012b}", codewords[i]);
            }
        }
    }
    return lexicodes.to_vec();
}

fn lexicode2(min_dist: u32, codewords: &Vec<u32>, lexicodes: &mut Vec<u32>) -> Vec<u32> {
    let clength = lexicodes.len();
    for i in 0..codewords.len() {
        if clength >= 1 {
            // let mut tmpvec: Vec<u32> = Vec::new();
            let mut flag: bool = true;
            for j in lexicodes.iter() {
                if codewords[i] <= *j {
                    flag = false;
                    continue;
                }
                // let indx = codewords.iter().position(|&x| x == *j).unwrap();
                if hamming_dist(*j, codewords[i]) < min_dist {
                    flag = false;
                    continue;
                }
            }
            if flag == true {
                lexicodes.push(codewords[i]);
                println!("adding: {:017b}", codewords[i]);
            }
        }
    }
    return lexicodes.to_vec();
}

fn main() {
    let codes: Vec<u32> = (0..64).collect();
    let bit_test = bitmask_test(&codes, 3);
    // for i in &codes {
    //     println!("{:08b}", i);
    // }
    // let test = codes[11] ^ codes[14];
    // let test2 = compat_matrix(&codes, 3);
    let mut codeinit: Vec<u32> = vec![0];
    // let codie_edit = lexicode(&mut codeinit, &test2);
    let codie_edit = lexicode3(5, &codes, &mut codeinit);
    for i in &codie_edit {
        println!("{:012b}", i);
        println!("int: {}", i);
    }
    // let codies = lexicode2(6, &codes, &mut codeinit);
    // for i in &codies {
    //     println!("{:017b}", i);
    //     println!("int: {}", i);
    // }
    println!("{}", codie_edit.len());
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
