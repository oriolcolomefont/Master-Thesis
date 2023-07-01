# Master Thesis in Sound and Music Computing
## Epidemic Sound AB & Universitat Pompeu Fabra (Music Technology Group)

<div style="display: flex; justify-content: space-between; align-items: center;">
  <img src="https://github.com/oriolcolomefont/Master-Thesis/blob/40562cf2514018c189965adab0da033aa8d7e021/Wordmark_2L_POS.png?raw=true" alt="alt text" width="300"/>
  <img src="https://github.com/oriolcolomefont/Master-Thesis/blob/54e35045debfb4f802cbc312afb681d6c41c7414/UPF-Logo.png?raw=true" alt="alt text" width="300"/>
</div>


### Title: Uncovering Underlying High-Level Musical Content in the Time Domain

Leveraging self-supervised deep neural networks, inductive bias, and aural skills to learn deep audio embeddings with applications to boundary detection tasks.

**Author:** Oriol Colomé Font

**Supervisors:** [Carl Thomé](https://github.com/carlthome) and [Carlos Lordelo](https://github.com/cpvlordelo)

**Date:** July 2023

## Abstract
This thesis posits the existence of invariant high-level musical concepts that persist regardless of changes in sonic qualities, akin to the symbolic domain where essence endures despite varying interpretations through different performances, instruments, and styles, among many other, almost countless variables.

In collaboration with Epidemic Sound AB and the Music Technology Group (MTG) at Universitat Pompeu Fabra (UPF), we used self-supervised contrastive learning to uncover the underlying structure of Western tonal music by learning deep audio features to improve unsupervised music boundary detection. We applied deep convolutional neural networks and a triplet loss function to identify abstract and semantic high-level musical elements without relying on their sonic qualities. This way, we replaced traditional acoustic features with deep audio embeddings, paving the way for sound-agnostic and content-sensitive music representation for boundary detection.

Our cognitively-based approach for learning embeddings focuses on using full-resolution data and preserving high-level musical information which unfolds in the time domain. Drawing upon our domain expertise, we developed robust transformations to encode heuristic musical concepts that should remain constant. This novel approach aims to reconcile music and machine learning, enhancing machine listening models' efficacy. 

Preliminary results suggest that, while not outperforming state-of-the-art results, our musically-informed technique has significant potential for boundary detection tasks and, most likely, nearly all MIR downstream tasks that are not purely sonic-based.

While music-motivated audio embeddings appear promising, delivering competitive results with room for improvement and potentially adaptable to other tasks constrained by data scarcity, the question remains if such general-purpose audio representation can mimic human hearing.

## Installation and Usage

*Detail the steps needed to install and run your project, if applicable*

### Requirements

*List out the software requirements for your project*

### Installation Steps

*Detail the steps to install the project*

### How to Use

*Provide steps on how to use the project*

## Results and Conclusion

*Provide an overview of your results and conclusions, if applicable*

## Acknowledgements

[Carl Thomé](https://github.com/carlthome) and [Carlos Lordelo](https://github.com/cpvlordelo) whose expertise in MIR, ML, and DSP has been pivotal to the progress and success of my thesis. Their invaluable guidance and unwavering encouragement have empowered me to overcome the challenges encountered throughout this journey. Their dedication and commitment to mentorship are truly inspiring, and their influence has left an indelible mark on my academic path. The opportunity to work under their mentorship has been a privilege and an honor. Their profound impact is beyond what words can encapsulate.

## License

*GNU General Public License v3.0*

## Contact

For any questions or concerns, please get in touch with Oriol Colomé Font at:
*oriolcolomefont@gmail.com*
*oriol.colome01@estudiant.upf.edu*
*oriol.colome.font@epidemicsound.com*
