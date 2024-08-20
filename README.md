# Crater Identification by Perspective Cone Alignment

The `environment.yml` file contains all the necessary libraries and the specific versions that the code was tested with.

An example of a testing data file is provided in the `data/` directory.

To run the code with the provided testing data and hyperparameters, execute the bash script:
```bash
bash run.sh
```

To use your own data, the following I/O adaptations are required:
 - Replace the calibration.txt file in the data/ directory with your own camera intrinsic parameters.
 - Replace the robbins_navigation_dataset.txt in the data/ directory with your own crater catalogue.
 - If you want to provide your own testing data, implement a custom I/O function to replace the testing_data_reading function (line 663) in PECAN.py. The minimum required input includes detected craters in ellipse form (x, y, a, b, theta) and the camera attitude.



This repository contains the code and data associated with the publication:

**Crater Identification by Perspective Cone Alignment**  
*Chee-Kheng Chng, Sofia Mcleod, Matthew Rodda, Tat-Jun Chin*  
Published in **Acta Astronautica**, Elsevier, 2024

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{chng2024crater,
  title={Crater identification by perspective cone alignment},
  author={Chng, Chee-Kheng and Mcleod, Sofia and Rodda, Matthew and Chin, Tat-Jun},
  journal={Acta Astronautica},
  year={2024},
  publisher={Elsevier}
}
