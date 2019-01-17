# RepresentationsLearningFromMulti_domain

Hi there, welcome!

This page contains the code and data used in the paper [Software Vulnerability Discovery via Learning Multi-domain Knowledge Bases].

### Idea:

Machine Learning (ML) based techniques for automated software vulnerability detection is a promising research direction. However, the performance of ML-based vulnerability detection approaches is far from satisifactory. Therefore manual detection still holds a dominent position. One of the main reasons is that there are insufficient high-quality labeled vulnerability data available for training a statistically robust classifier (in supervised detection scenario). In this paper, we proposed an idea to utilize multiple vulnerability-relevent data sources to compensate for the shortage of the labeled data and facilitate the learning of high-level representations that are indicative of vulnerable function patterns. 

The vulnerability-relevent data sources refer to the available labeled data provided by other studies and the artificially constructed vulnerability test cases from the [the Software Assurance Reference Dataset (SARD) project](https://samate.nist.gov/SRD/). We propose a framework to bridge the differences between the data from the two sources, and to learn function-level representations as effective features without the need of tedious manual feature engineering process. Besides, the framework can be extended if more data sources are available. The experiments demonstrated the effectiveness of our proposed framework.

### About this repository:

The "Code" folder contains the Python code for:
1. Load the source code functions -- LoadCFilesAsText.py
2. Train Bi-LSTM model -- BiLSTM_model.py and Train_BiLSTM.py
3. Obtain representations -- ObtainRepresentations.py
4. Use the representations as features for training and test -- testRandomForest.py

The "Data" folder contains a small amount of data samples (more data will be shared when the paper is accepted.):
1. The extracted serialized Abstract Syntax Tree (AST) sequence samples from project FFmpeg -- stored in the "The_CVE_samples" folder. 
2. The source code of the SARD project test case and the obfuscated version -- stored in the "The_SARD_samples" folder.

### Reproduction:

1. Software Environment

 * [Tensorflow](https://www.tensorflow.org/) >= 1.5
 * [Keras](https://github.com/fchollet/keras/tree/master/keras) >= 2.1.6
 * [Scikit-learn](http://scikit-learn.org/stable/)
 * [Gensim](https://radimrehurek.com/gensim/)
 * Python >= 2.7
 * [CuDNN](https://developer.nvidia.com/cudnn) >=7.1.4
 
The dependencies can be installed using [Anaconda](https://www.anaconda.com/download/). For example:

```bash
$ bash Anaconda3-5.3.1-Linux-x86_64.sh
```
 
2. Hardware
 * An NVIDIA GPU with as least 4G video memory is required (Due to using CuDNNLSTM, training on a mainstream GPU had a 12x ~ 50x speedup compared with training on a server equipped with two high-end Intel Xeon CPUs with totally 48 logical cores on our data set). 
 
3. Data preprocessing
 * The AST parser -- The AST of a source code function can be extracted using the tool called [CodeSensor](https://github.com/fabsx00/codesensor). In our paper, we used the old version which is codeSensor-0.2.jar. The parser does not require a build environment/supporting libraries for parsering the source code functions to ASTs. 
 * The obfuscation tool we used for obfuscating the SARD project samples is call [Snob](https://snob.soft112.com/). 

Thank you! 
