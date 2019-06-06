# Voice Activity Detection

## Signal 

![Screen Shot 2019-05-10 at 11.01.07 am](/Users/jessicaxu/Desktop/Screen Shot 2019-05-10 at 11.01.07 am.png)

### Signal-to-Noise Ratio (SNR)

- A measure to evaluate the level of the desired voice to the level of the background noise

- A ratio of the signal power to the noise power

-  Expressed in decibles

- Higher than 1:1: more signal than noise

-  A reference for VAD evaluation: deciding the features that should be used 

![Screen Shot 2019-05-10 at 11.07.06 am](/Users/jessicaxu/Desktop/Screen Shot 2019-05-10 at 11.07.06 am.png)

### Signal analysis

- Customary divide a signal into slide window or just singular window

  ![Screen Shot 2019-05-10 at 11.09.31 am](/Users/jessicaxu/Desktop/Screen Shot 2019-05-10 at 11.09.31 am.png)

- Window:

  — The width of the window will impact the precision of the output

  — Wider width: the better accuracy on the frequency scale

  — Narrower width: the better precision on the time scale

  — Overlapping window: the mitigation of side effect for window straddling on both speech and non-speech frames

-  Mainly use window of size 1024 (0.064s at 16000Hz) or 2048 (0.128s) as they are powers of 2 with overlapping equal to half the window

## Features

### Short-Term Energy (STE)

- The most commonly used feature to discriminate the voice activity from non-voice activity.

- Rely on the average absolute amplitude on the windows

  ![img](https://cdn-images-1.medium.com/max/1440/1*_Sq-x7BNwozGFNZhpGQREQ.png)

- Relation with SNR: 

  — STE is efficient for high SNR

  — Ineffective when SNR < 1

- discriminate speech from noises like impact noise, which are as loud or louder than human voice

###  Fourrier Transform( Frequency domain)

- Decomposes a signal into the frequencies(spectrum)

  — Given audio window, it returns the frequency distribution

  ![Screen Shot 2019-05-10 at 11.30.13 am](/Users/jessicaxu/Desktop/Screen Shot 2019-05-10 at 11.30.13 am.png)

### Spectrogram

- The spectrum over time

- Frequencies of human voice can be comply to a pattern

  ![img](https://cdn-images-1.medium.com/max/1440/1*PvAtg8RgVD__k-_N7-5o3g.png)

### Spectral Flatness Measure(SFM)

- A measure of the flatness of the spectrum
- A ratio of the geometric mean over the arithmetic mean
- Computes if the signal is noise-like speech (tend to 1) or tone-like speech(tend to 0) 

![img](https://cdn-images-1.medium.com/max/1440/1*I2ilH7d1t_SvGkAx_qDNpA.png)

![img](https://cdn-images-1.medium.com/max/1800/1*bFTg1L1pFojCvgl_LBPnpg.png)

### Dominant frequency

- Seeking the frequency( *Fundamental Frequency*) which has the higher spectrum amplitude

- Based on the fact that human voice has  specific and known funcdamental frequencies (pitch)

- Fundamental frequency 

  — Male: 80Hz~180Hz

  — Famale: 160Hz~260Hz

  — Children: around 260Hz

  — Crying baby: 500Hz

![img](https://cdn-images-1.medium.com/max/1440/1*LI2Nv7-JhpiQzFvO2xP6jw.png)

### Spectrum Frequency Band Ratio

- Dominant Frequency + SFM

- A ration of a specific frequency band amplitudes over the whole spectrum

- The band's border have been obtained through experimentation and set to [80Hz-1000Hz]

  ![img](https://cdn-images-1.medium.com/max/1440/1*hrFkWYNlcZIQonWrPDC_9Q.png)

### MFCC, FBANK, PLP

![img](https://cdn-images-1.medium.com/max/1440/1*8z2m14FjrAdcFZQII7EiZQ.png)

- Reduce and compress the number of infromation by keep the most relavant ones
- Retrun an array of values in opposition to the previous features

### Treshold

1. Type

   - stactic treshold

     — set once

     — For known non-speech frequency: Beginning of the decoding on the new window

     — For frequency-wise features: Outside of context for feature

   - dynamic treshold

     — adjust over time

     — accommodate changes

     — adjust at every window to be higher than one classified as silence, and  lower than one classified as speech.

     — compute the pondered mean between the mean of last x window of silence and the mean of the last x window of speech.

2. Consideration:

   - value of treshold 
   - keep context variation
   - number of features

3. A window will be decided by the fact that the feature for the window is higher or lower than the treshold 

# VAD Models

## DNN-based VADs

1. Model 1
   - Combining several acounstic feature as DNN inputs and determines the useful ones
2. Model 2:
   - Data-driven acoustic features directly from raw speech (or a spectrogram) as input
   - Use CNN to classify speech/non-speech parts
   - Feature extraction in CNN

3. Model 3 :
   - focusing on effecitve utilization
   - The context information(CI) from DNN
   - The duration of the speech signals
   - LSTM/RNN-based
     1.  encode long-short-term CI from inputs features
     2. Slower than DNN, as of the sequentiality 
     3. Couldn't outperform vellina DNN-based(model 1)
   - Boosted DNN (bDNN)
     1. Inputs/ouputs CI are exploted by applying multiple inputs/ouput units on DNN
     2. Based on a fully-connected Neural Network (FNN) with a fixed size of inputs and outputs

## Adaptive Context Attention Model (ACAM) based VAD

— Building model: attention with DNN

— Training model: reinforcement learning with effectie context attention process

### Overview

- Frame(window)-based speech/noise classifier

- input speech signal is divide into overlapping 25-ms frames with 10-ms shifts (sliding window)

- Acoustic feature vectors are extracted from each frame

- Notation

  1. Dataset: $ \{ (x_m,y_m^{truth})\}_{m=1}^M​$ where,

     ​	     —  $m=1,…,M​$ : the frame index

     ​	     — $x_m \in R^D$: acoustic feature vector for frame $m$

     ​	     —  $y_m^{truth} \in [0,1]:$ the label for $x_m$

  2. Input/output CI: $ \{ (v_m,v_m^{truth})\}_{m=1}^M$, where,

     ​	     — selecting neibouring frames to expand data

     ​	     — flattening data:		  $V_m=[X_{m-W}^T,X_{m-W+u}^T,…,X_{m-1-u}^T,X_m^T,,X_{m-1}^T,X_m^T,X_{m+1}^T,X_{m+1+u}^T,…,X_{m+W-u}^T,X_{m+W}^T]^T​$

      $y_m^{truth}=[y_{m-W}^{truth},y_{m-W+u}^{truth},…,y_{m-1-u}^T,X_m^T,,X_{m-1}^{truth},y_m^{truth},y_{m+1}^{truth},y_{m+1+u}^{truth},…,y_{m+W-u}^{truth},y_{m+W}^{truth}]^T$

  3. $W, u$ are user-defined parameter to fine tuned

### Model Architecture (RNN+attention)

![Screen Shot 2019-05-10 at 2.42.46 pm](/Users/jessicaxu/Desktop/Screen Shot 2019-05-10 at 2.42.46 pm.png)

#### Decoder

- Initialise: internal state at the time step 1 —>  $h_{m_0},t=1$

- input feed: internal state from the previous time step —> $h_{m_{t-1}}$

- Output: attention input feed, which frames in $v_m$ should be focused on

- Affine transformation: $affine(x |\theta)=W*x+b$

- Activate function: *RELU* for the non-linear transformation

- Output transformation: *SIGMOID* $\sigma (x)=\frac{1}{1+exp(-x)}$

- Smoothed softmax classifier (3)

  — sharpened the attention

  — degrading the CI deployment 

  — reducing negatively affecting VAD performance from sigmoid function

- Final notation:

  $ A'_{m_t}=RELU(Affine(h_{m_{t-1}}|\theta_{attention}))$

  ​       $= [a'_{m_t,1},……,a'_{m_t,k},……,a'_{m_t,L}]^T……………………………(1)$

  $a_{m,t,k}=$ 



#### Attention

- 

​	











