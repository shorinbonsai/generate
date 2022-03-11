use std::collections::HashSet;
use std::iter::FromIterator;
use std::ops::Index;

use rand::seq::SliceRandom;
use rand::Rng;
use rand::RngCore;

pub struct GeneticAlgorithm<S> {
    selection: S,
}

impl<S> GeneticAlgorithm<S>
where
    S: SelectionMethod,
{
    pub fn new(selection: S) -> Self {
        Self { selection }
    }

    pub fn evolve<I>(&self, rng: &mut dyn RngCore, population: &[I]) -> Vec<I>
    where
        I: Individual,
    {
        assert!(!population.is_empty());
        (0..population.len())
            .map(|_| {
                let parent_1 = self.selection.select(rng, population).chromosome();
                let parent_2 = self.selection.select(rng, population).chromosome();
                // TODO crossover
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
    ) -> Chromosome {
        assert_eq!(parent_1.len(), parent_2.len());
        let mut child = Vec::new();
        let gene_count = parent_1.len();

        for gene_idx in 0..gene_count {
            let gene = if rng.gen_bool(0.5) {
                parent_1[gene_idx]
            } else {
                parent_2[gene_idx]
            };

            child.push(gene);
        }

        child.into_iter().collect()
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

// #[derive(Clone, Debug, Default)]
// pub struct TournamentSelection;

// impl SelectionMethod for TournamentSelection {
//     fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I], t_size: usize)
//     where
//         I: Individual,
//     {
//         assert!(!population.is_empty());
//         let tournament = population.choose_multiple(rng, t_size);
//         let mut parents = vec![];
//         for i in tournament {
//             if parents.len() < 2 {
//                 parents.push(i);
//             }
//             parents.sort_by(|a, b| b.fitness().cmp(&a.fitness()));
//             for j in parents {
//                 if i.fitness() > j.fitness() {
//                     let t = parents.remove(index);
//                 }
//             }
//         }
//     }
// }
