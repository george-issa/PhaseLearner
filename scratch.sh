  conda deactivate                                                                                                                                            
  conda env remove -n phaselearner -y                                                                                                                         
  conda env create -f environment.yml                                                                                                                         
  conda activate phaselearner                                                                                                                                 
  pip install -e .          # editable install of ssh_ml_phase
