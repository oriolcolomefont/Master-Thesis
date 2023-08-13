# Master Thesis in Sound and Music Computing
## Epidemic Sound AB & Universitat Pompeu Fabra (Music Technology Group)

<img src="https://github.com/oriolcolomefont/Master-Thesis/blob/3477a79ff7821c1d296068c376b8afb854b7f092/Epidemic_Sound_Logo_White.png?raw=true" alt="alt text" width="200"/>
<img src="https://github.com/oriolcolomefont/Master-Thesis/blob/54e35045debfb4f802cbc312afb681d6c41c7414/UPF-Logo.png?raw=true" alt="alt text" width="200"/>

### Uncovering Underlying High-Level Musical Content in the Time Domain

Leveraging self-supervised deep neural networks, inductive bias, and aural skills to learn deep audio embeddings with applications to boundary detection tasks.

**Author:** Oriol Colomé Font

**Supervisors:** [Carl Thomé](https://github.com/carlthome) and [Carlos Lordelo](https://github.com/cpvlordelo)

**Date:** July 2023

## Abstract

This thesis posits the existence of invariant high-level musical concepts that persist regardless of changes in sonic qualities, akin to the symbolic domain where essence endures despite varying interpretations through different performances, instruments, and styles, among many other, almost countless variables.

### Collaboration and Methodology

In collaboration with Epidemic Sound AB and the Music Technology Group (MTG) at Universitat Pompeu Fabra (UPF), we used self-supervised contrastive learning to uncover the underlying structure of Western tonal music by learning deep audio features to improve unsupervised music boundary detection. 

We applied deep convolutional neural networks and a triplet loss function to identify abstract and semantic high-level musical elements without relying on their sonic qualities. In doing so, we replaced traditional acoustic features with deep audio embeddings, paving the way for sound-agnostic and content-sensitive music representation for boundary detection.

### Approach

Our cognitively-based approach for learning embeddings focuses on using full-resolution data and preserving high-level musical information which unfolds in the time domain. A key component in our methodology is the use of triplet networks, which provide an effective way of understanding and preserving the relative distances between different pieces of music. 

This model structure is integral to our work, allowing us to identify and maintain the nuanced relationships within our musical data. Drawing upon our domain expertise, we developed robust transformations to encode heuristic musical concepts that should remain constant. This novel approach aims to reconcile music and machine learning, enhancing machine listening models’ efficacy through deep learning and triplet networks.

### Preliminary Results and Future Considerations

Preliminary results suggest that, while not outperforming state-of-the-art results, our musically-informed technique has significant potential for boundary detection tasks and, most likely, nearly all MIR downstream tasks that are not purely sonic-based.

While music-motivated audio embeddings don't outperform state-of-the-art results, they appear promising, delivering competitive results with room for improvement and potentially adaptable to other tasks constrained by data scarcity; the question remains if such general-purpose audio representation can mimic human hearing.

### Keywords

- MIR, Music Structure Analysis, Deep audio embeddings, Audio representations, Representation learning, Embeddings, Transfer learning, Multi-task learning, Multi-modal learning, Aural Skills


## Installation and Usage

### Pip

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.

`pip install -r requirements.txt`

Alternatively, you can use [conda](https://docs.conda.io/en/latest/) to create a virtual environment and install the required dependencies.

`conda create -n project-env python=3.8`
`conda activate project-env`
`conda install --file requirements.txt`

### How to Use

*Provide steps on how to use the project*

## Results and Conclusion

*Provide an overview of your results and conclusions, if applicable*

## Acknowledgements

[Carl Thomé](https://github.com/carlthome) and [Carlos Lordelo](https://github.com/cpvlordelo), whose unrivaled expertise in the areas of MIR, ML, and DSP has not only been pivotal to the success of my thesis but their wisdom and guidance have been a constant source of motivation and enlightenment. Their genuine enthusiasm for mentorship and an intrinsic knack for going above and beyond has been truly inspiring. Throughout my journey, they have been more than just mentors; they have become the embodiment of academic kindness and professionalism. They never made me feel inferior or judged but treated me as an equal peer, fostering an environment of respect and intellectual growth.

Their unwavering support and encouragement have empowered me to tackle and overcome the myriad challenges encountered during this journey. Their influence has left a lasting mark on my academic path and their mentorship, a privilege I will forever hold in high esteem. The profundity of their impact, which extends well beyond the scope of this thesis, is not easily encapsulated by mere words but is deeply felt in the better researcher I have become under their guidance.

## License

*GNU General Public License v3.0*

## Contact

For any questions or concerns, please get in touch with Oriol Colomé Font at: *oriolcolomefont@gmail.com*, *oriol.colome01@estudiant.upf.edu*, or *oriol.colome.font@epidemicsound.com*
