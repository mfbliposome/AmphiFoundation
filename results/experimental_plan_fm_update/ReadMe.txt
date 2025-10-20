In these folder, it has 2 sets of experimental plan:

In the 'region' folder, it contains the experimental plan based selecting samples from different regions on latent space of foundation model.
It contains 3 subfolder: 'concentrations', 'dispense_volume', and 'integrate', 
which include files for concentration of amphiphiles, dispense volumes and combined dispense volume file for same sample.
Here's naming system for these files (e.g. high_original_fm_original_concentrations.csv, low_replace_dissmi1_fm_original_concentrations.csv):  
'low' or 'high' stands for predicted probability from model;
'original' are samples from original amphiphiles;
'replace_dissmi1' are samples from dataset where 1 amphiphile was replaced by a new dissimilar amphiphile...etc.


In the 'special' folder, it contains the experimental plan based on predict probability difference when replace amphiphile and changing concentration.
It contains 3 subfolder: 'concentrations', 'dispense_volume', and 'integrate',
which include files for concentration of amphiphiles, dispense volumes and combined dispense volume file for all samples.
Here's naming system for these files (e.g. sample144_concentration_ori_fm_dispense_volumes.csv, sample144_concentration_smi_fm_dispense_volumes.csv):  
'sample144' stands for sample index from Active learning paper,
'concentration_ori' stands for amphiphiles are not replaced
'concentration_smi(dissmi)' stands for one amphiphile replaced by similar(dissimilar) amphiphiles.
