# AmphiFoundation
Foundation model for MFB liposome

## Folder structure
In this repo, it has 4 folders: `data/`, `code/`,  `envs/` and `results/`.

### Folder `data/`
In this folder, it stores the data files for the foundation model related analysis and microscopy images:
- `Intermediate/`

  Contains intermediate data files and model files

- `Literature/`

  Contains the data from literature and it has 3 subfolders: `surfactant_paper`, `binary_paper`, `amphiphile_paper`

- `Tempplates_microscopy/`

  Contains the templates that are used for microscopy image analysis. 

  See details in here: [[Data description](https://github.com/hliu56/AmphiFoundation/blob/main/data/README.md)]

-  Original microscopy image data and vesicles detection results can be seen in [[Zenodo link](https://doi.org/10.5281/zenodo.17401465)]

Note, in this repository, we have used the terms 'similar' and 'dissimilar' as a shorthand for hydrocarbon and fluorocarbon amphiphiles. However, in the manuscript, we have opted to use the formal name, hydrocarbon and fluorocarbon amphiphiles, to elucidate their chemical structures.



### Folder `code/`
- In this folder, it has 2 subfolders: `src/`, `notebooks/`, which store the source code for the foundation model related analysis, microscopy analysis, and working notebooks.
- The `src/` folder contains the source code throughout the study. Each module corresponds to a specific functionality used in latent space establishment, model comparison, analysis, etc.
  1. The subfolder `models/` contains the pretrained foundation models.
  2. The subfolder `microscopy/` contains the source code for microscopy analysis.
     
- The `notebooks/` folder contains working notebooks throughout the study. The notebook name reflects its main purpose and corresponds to the analysis and figures presented in the paper.

  To help navigation, notebooks are organized into 4 logical groups:
  1. Latent space establishment (Latent_space_*.ipynb)
     These notebooks generate molecular representations (embeddings) using different foundation models or fingerprints. For example:
     - Latent_space_smi-ted_*.ipynb — latent space generation using the SMI-TED foundation model.
     - Latent_space_cheprop_*.ipynb — latent space generation using the Chemprop foundation model.

  2. Model training and comparison (Model_comparison*.ipynb)
     Evaluate and compare the performance of different models trained with various representations.

  3. Experimental planning and fine-tuning (FM_ExperimentalPlan*.ipynb, FM_finetune*.ipynb)
     Notebooks document the experimental design, microscopy analysis, fine-tuning, etc.

  4. Concentration effect analysis (FM_Concentration_affect*.ipynb)
     These notebooks analyze how varying amphiphile concentrations affect model predictions and vesicle formation.
     
### Folder `envs/`
This folder contains the Python environment specifications required to reproduce the computational experiments.
It includes the base environment used for general analysis and additional environments for different foundation models.




### Folder `results/`
This folder contains the results produced by working notebooks.




## Software requirements and installation
- Base environment
   
   Used for data processing, training, and evaluation of machine learning models. The detailed environment dependencies can be seen in here: `envs/base_environment.yml`.
   The installation can be done with:
   ```
   conda env create -f envs/base_environment.yml
   conda activate base_env
   ```
- Foundation model environments
   
   The repository integrates several Foundation Models (FMs) developed by other research groups.
   Each model requires its own environment, following the original installation guide from their official repositories.
   Environment dependencies are listed in the `envs/` folder for reference.
   
  | Model | Description | Source | Installation |
|--------|--------------|---------|---------------|
| **VICGAE** |  Gated recurrent autoencoder. Used to generate molecular latent space. | [VICGAE GitHub]( https://github.com/HumbleSituation164/orion-kl-ml-main) | See the setup instructions in the original repo: [VICGAE documentation](https://github.com/HumbleSituation164/orion-kl-ml-main/blob/main/README.md). |
| **Chemprop** | Directed message-passing neural networks (D-MPNNs). Used to generate molecular latent space. | [Chemprop GitHub](https://github.com/chemprop/chemprop) | Follow installation guide in the [Chemprop documentation](https://chemprop.readthedocs.io/en/main/installation.html). |
| **CheMeleon** | Based on the Chemprop architecture. Used to generate molecular latent space. | [CheMeleon Github](https://github.com/JacksonBurns/chemeleon) | See the setup instructions in the original repo: [CheMeleon documentation](https://github.com/JacksonBurns/chemeleon/blob/main/README.md).|
| **SMI-TED** |  Transformer-based foundation model. Used to generate molecular latent space | [SMI-TED Github](https://github.com/IBM/materials) | See the setup instructions in the original repo: [SMI-TED documentation](https://github.com/IBM/materials/blob/main/README.md). |


## Citation

If you use this repository in your research, please cite it as follows: 
TBD

## License

This project is licensed under the [BSD 3-Clause License](LICENSE). Please cite this repository if you use it in your work.



