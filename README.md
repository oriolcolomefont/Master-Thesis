# [Master Thesis in Sound and Music Computing](https://zenodo.org/records/8380670)
## Epidemic Sound AB & Universitat Pompeu Fabra (Music Technology Group)

### Uncovering Underlying High-Level Musical Content in the Time Domain

Leveraging self-supervised deep neural networks, inductive bias, and aural skills to learn deep audio embeddings with applications to boundary detection tasks.

**Author:** Oriol Colomé Font

**Supervisors:** [Carl Thomé](https://github.com/carlthome) and [Carlos Lordelo](https://github.com/cpvlordelo)

**Date:** July 2023

## Abstract

The thesis explores invariant high-level musical concepts that persist despite changes in sonic qualities. Collaborating with Epidemic Sound AB and the Music Technology Group at Universitat Pompeu Fabra, the study employs self-supervised contrastive learning to uncover the structure of Western tonal music using deep audio features. The approach utilizes triplet networks and focuses on full-resolution data to preserve high-level musical information over time. The study aims to create sound-agnostic and content-sensitive music representations for track segmentation tasks by replacing traditional acoustic features with deep audio embeddings. Preliminary results show potential for boundary detection tasks, suggesting effectiveness in sound-agnostic and content-sensitive applications with data scarcity. The question posed is whether this general-purpose audio representation can replicate human hearing.

### Collaboration and Methodology

In collaboration with Epidemic Sound AB and the Music Technology Group (MTG) at Universitat Pompeu Fabra (UPF), we used self-supervised contrastive learning to uncover the underlying structure of Western tonal music. Our approach involved learning deep audio features to improve unsupervised music boundary detection.

### Approach

Our cognitively-based approach for learning embeddings focuses on using full-resolution data and preserving high-level musical information that unfolds in the time domain. A key component in our methodology is the use of triplet networks, which provide an effective way of understanding and preserving the relative distances between different pieces of music. 

This model structure is integral to our work, allowing us to identify and maintain the nuanced relationships within our musical data. Drawing upon our domain expertise, we developed robust transformations to encode heuristic musical concepts that should remain constant. This novel approach aims to reconcile music and machine learning, enhancing machine listening models’ efficacy through deep learning and triplet networks.

### Preliminary Results and Future Considerations

Preliminary results suggest that, while not outperforming state-of-the-art results, our musically-informed technique has significant potential for boundary detection tasks and, most likely, nearly all MIR downstream tasks that are not purely sonic-based.

While music-motivated audio embeddings don't outperform state-of-the-art results, they appear promising, delivering competitive results with room for improvement and potentially adaptable to other tasks constrained by data scarcity. The question remains if such a general-purpose audio representation can mimic human hearing.

### Keywords

- MIR
- Music Structure Analysis
- Deep audio embeddings
- Audio representations
- Representation learning
- Embeddings
- Transfer learning
- Multi-task learning
- Multi-modal learning
- Aural Skills

## Virtual environment installation

### Pip

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.

<pre>
pip install -r requirements.txt
</pre>

Alternatively, you can use conda to create a virtual environment and install the required dependencies.

<pre>
conda create -n project-env python=3.8
conda activate project-env
conda install --file requirements.txt
</pre>

## Acknowledgements

[Carl Thomé](https://github.com/carlthome) and [Carlos Lordelo](https://github.com/cpvlordelo), whose unrivaled expertise in the areas of MIR, ML, and DSP has not only been pivotal to the success of my thesis but their wisdom and guidance have been a constant source of motivation and enlightenment.

## License

*GNU General Public License v3.0*

## Contact

For any questions or concerns, please get in touch with Oriol Colomé Font at:
- *oriolcolomefont@gmail.com*
- *oriol.colome01@estudiant.upf.edu*
