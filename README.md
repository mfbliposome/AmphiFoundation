# AmphiFoundation
Foundation model for MFB liposome

## Folder structure
In this repo, it has 3 folders: 'code', 'data', and 'results'.

### folder 'code'
- In this folder, it has 2 subfolders: 'src', 'notebooks', which store the source code for the foundation model related analysis, microscopy analysis, and working notebooks.

### folder 'data'
In this folder, it stores the data files for foundation model related analysis and microscopy images:
-  input data surfactant, binary aqueous mixture, and amphiphile mixture
-  Original microscopy image and vesicles detection results. These data files can be seen in [[Zenodo link](https://doi.org/10.5281/zenodo.17401465)]

Note, in this repository, we have used the term 'similar' and 'dissimilar' as a shorthand for hydrocarbon and fluorocarbon amphiphiles. However, in the manuscript, we have opted to use the formal name, hydrocarbon and fluorocarbon amphiphiles, to elucidate their chemical structures.

### folder 'results'
In this folder, it contains the results produced by working notebooks.

## Software requirements and installation
1. Base environment
   Used for data processing, training, and evaluation of machine learning models. The detailed environment dependencies can be seen in here: `envs/base_environment.yml`.
   The installation can be done with:
   ```
   conda env create -f envs/base_environment.yml
   conda activate base_env
   ```
2. Foundation model environments
   The repository integrates several Foundation Models (FMs) developed by other research groups.
   Each model requires its own environment, following the original installation guide from their official repositories.
   Environment dependencies are listed in the `envs/` folder for reference.
   
   | Model          | Description                                                                                    | Source                                                  | Installation                                                                                                  |
| -------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Chemprop**   | Graph neural network for molecular property prediction. Used to generate molecular embeddings. | [Chemprop GitHub](https://github.com/chemprop/chemprop) | Follow installation guide in the [Chemprop documentation](https://github.com/chemprop/chemprop#installation). |
| **Other FM 1** | (e.g., text-based FM for chemical captions)                                                    | [GitHub Link]                                           | See the setup instructions in the original repo.                                                              |
| **Other FM 2** | (e.g., image encoder for microscopy data)                                                      | [GitHub Link]                                           | Follow their installation guide.                                                                              |


   
- numpy
- scipy
- pandas
- matplotlib
- scikit-learn
- opencv

## Citation

If you use this repository in your research, please cite it as follows: 
TBD

## License

This project is licensed under the [BSD 3-Clause License](LICENSE). Please cite this repository if you use it in your work.



