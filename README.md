# Angle recognition 

Master's thesis project - Condensed matter physics problems and deep learning solutions 

## Description

The main idea lies in leveraging deep learning techniques to recognize the shift angles between layers of Van der Waals heterostructures. In our specific case, the material in question is Nanoporous Graphene (NPG).
Scanning Tunneling Microscopy (STM) is employed for imaging the graphene
layers, thus, enabling the detection of twisted bilayer and multilayer domains. On the simulation side, an analytic model is adopted to shed light on the number of misoriented layers, and their weak
interaction: for each layer we can write an accurate analytical form that can describe the in-plane density.

## Getting Started

## Step 1: Physical Model and Image Simulation

The first step directly concerns the physics of the problem: the analytical model suggests that the in-plane electronic density of a graphene layer is given by 

![In plane density equation](immages_md/in_plane_density.png)

while the full three-dimensional electronic density is modeled as

![Tridimensional density equation](immages_md/3d_density.png)

The first notebook, `npg_stm_images.ipynb`, is used to simulate STM-like images for a tuple of randomly generated twist angles, which depend on the number of layer: one acquired in Constant Current Mode (CCM-direct lattice), its corresponding image in reciprocal space (CCM-reciprocal), and one acquired in Constant Height Mode (CHM-direct).

## Step 2: Dataset Generation

Now that we have all the functions needed to generate a set of images for a single tuple of angles, we need to create a sufficient number of such tuples to build a dataset suitable for training a neural network. 

The `generate_dataset.py` script is designed to be adaptive, with all key variables defined via argument parsing to accommodate different generation environments. In my case, I had access to a CPU-based SLURM cluster, which allowed me to parallelize the generation process according to the number of jobs I could submit.

In parallel with the image generation, a metadata file is created to associate each image set with its corresponding angle tuple and identifier. Everything is then packaged into a single `.HDF5` file.
### Second step: CNN classifier

* How/where to download your program
* Any modifications needed to be made to files/folders

<!-- ### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
``` -->

## Authors

Contributors names and contact info

MSc Giuseppina Pia Varano 
[@GiusyVarano](https://www.linkedin.com/in/giusy-varano-0277202aa/)

## Version History

* 0.2
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release 

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
<!-- * [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46) -->
