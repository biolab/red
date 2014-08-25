Red
===========

Epistasis analysis is a major tool from classical genetics to infer the order of function of genes in a common pathway. Commonly, it considers single and double mutant phenotypes and for a pair of genes observes if change in one masks the effects of the other one. Despite recent emergence of biotechnology techniques that can provide gene interaction data on a large, possibly genomic scale, very few methods are available for quantitative epistasis analysis.

Red is a conceptually new probabilistic approach to gene network inference from quantitative interaction data. The advantage of Red is the global treatment of the phenotype data through a factorized model and probabilistic scoring of pairwise gene relationships from latent gene presentations. Resulting gene network is assembled from scored relationships.

This repository contains supplementary material for *Gene network inference by probabilistic scoring of relationships from a factorized model of interactions*,
presented at ISMB 2014 by Marinka Zitnik and Blaz Zupan.

Usage 
-----
	
Inferring preferential order-of-action factorized model with default parameters::

	>>> from red import Red
	>>> from data import loader
	>>> G, S, H, genes = loader.load_jonikas_data("data/080930a_DM_data.mat")
	>>> gene_red = Red(G, S, H, genes)
	>>> gene_red.order()

For more examples see ``examples.py`` or run::

    $ python examples.py
    
Input:
    * matrix of double mutant phenotypes, ``G``,
    * matrix of expected no-interaction double mutant phenotypes, ``H``,
    * a vector of single mutant phenotypes, ``S``,
    * latent dimension, ``rank``,
    * regularization of gene latent representation, ``lambda_u``, ``lambda_v``,
    * learning rate of gene latent profiles, ``alpha``,
    * learning rate of logistic map, ``beta``.

Inferred factorized model ``gene_red`` includes:
	* preferential order-of-action scores,
	* completed matrix of double mutant phenotypes,
	* gene-dependent logistic function model,
	* inferred gene network for a given gene set of interest,
	* gene latent representation,
	* quality (Fro. error and NRMSE) of matrix completion.

Citing
------

	.. code-block:: none

        @article{Zitnik2014,
          title={Gene network inference by probabilistic scoring of relationships from a factorized model of interactions},
          author={{\v{Z}}itnik, Marinka and Zupan, Bla{\v{z}}},
          journal={Bioinformatics},
          volume={30},
          number={12},
          pages={i246--i254},
          year={2014},
          publisher={Oxford University Press}
        }


Contact: marinka.zitnik AT fri.uni-lj.si