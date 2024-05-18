import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit

class Neural_Network:

    def __init__(self, INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE):

        self.input_nodes = INPUT_NODES
        self.hidden_nodes = HIDDEN_NODES
        self.output_nodes = OUTPUT_NODES

        self.learning_rate = LEARNING_RATE

        self.activtion_function = lambda x : expit(x)

        self.inverse_activation_function = lambda x : logit(x)

        self.weigths_input_to_hidden_layer = np.random.default_rng().normal(0, pow(self.input_nodes, -0.5),
                                                          (self.hidden_nodes, self.input_nodes))
        self.weigths_hidden_to_output_layer = np.random.default_rng().normal(0, pow(self.hidden_nodes, -0.5),
                                                          (self.output_nodes, self.hidden_nodes))

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T

        x_hiddem = np.dot(self.weigths_input_to_hidden_layer, inputs)
        o_hidden = self.activtion_function(x_hiddem)

        x_output = np.dot(self.weigths_hidden_to_output_layer, o_hidden)
        o_output = self.activtion_function(x_output)

        return o_output


learning_rate = [0.01, 0.1, 0.2, 0.3, 0.6]
performance_lr = [94.78, 97.21, 96.86, 96.24000000000001, 91.36]

epoch = [1, 2, 3, 4, 5, 7, 10, 20]
performance_e = [95.72, 96.61999999999999, 97.03, 97.42, 97.34, 97.52, 97.53, 97.21]

nodes = [10, 100, 200, 250, 500, 1000]
performance_n = [89.3, 96.76, 97.24000000000001, 97.28, 97.5, 97.55]

ls = ["Main","Links","Optimization", "Brain Scan", "Test NN"]
option = 'Main'

option = st.selectbox('Switch Slide:', ls)

st.title(option.upper())


if option == ls[0]:

    st.write('-------')

    st.write('''
             # Neural Net - MNIST
        It is a university project and an exercise for neural networks.

        Data is obtained from this [website](https://pjreddie.com/projects/mnist-in-csv/) in CSV format. 
        The activation function is **sigmoid**, and the weight distribution has been derived from the following relationships:
        $\pm1/\sqrt{incomin Links}$ A normal distribution where the mean is zero.

        ## Model Accuracy and Optimization
        The model has achieved an accuracy of **97.53%** with a learning rate of **0.07**, over **5 epochs**, and **600 nodes**. It exhibits the lowest accuracy of approximately **95%** on digits 7 and 2, while the highest accuracy is around **99%** on digits 1 and 0. 

        For new data, the model requires optimization as it has been trained. It currently struggles with noisy images.

        ### Note on Optimization
        To maintain the integrity of the model's performance on new datasets, it is crucial to optimize it following the established training protocols.


        ## Requirement
        - **matplotlib**==3.8.4
        - **imageio**==2.34.1
        - **numpy**==1.26.4
        - **scipy**==1.13.0

        **[Github](https://github.com/HamidHekmatnezhad/NN_MNIST)**     
        **[Artikel - in persian language](https://docs.google.com/document/d/1_BfeoZNyo_W1c6rmdG1JzUO3GuJJeaw0KLrDfq47dwc/edit?usp=sharing)**
''')
    
elif option == ls[1]:

    col1, col2 = st.columns(2)

    col1.markdown("""
    # [Github](https://github.com/HamidHekmatnezhad/NN_MNIST)
                  """)
    
    col2.markdown("""
    # [Artikel](https://docs.google.com/document/d/1_BfeoZNyo_W1c6rmdG1JzUO3GuJJeaw0KLrDfq47dwc/edit?usp=sharing)
                  """)

elif option == ls[2]:

    st.write('-------')

    plt.style.use('dark_background')
    fig, ax = plt.subplots(3,1)
    ax[0].plot(learning_rate, performance_lr, linestyle='--', marker='o', color='y')
    ax[0].set_ylabel('Learning Rates')
    ax[1].plot(epoch, performance_e, marker='o', linestyle='--', color='r')
    ax[1].set_ylabel('Epochs')
    ax[2].plot(nodes, performance_n, marker='o', linestyle='--', color='b')
    ax[2].set_ylabel('Nodes')

    st.pyplot(fig)

    st.markdown("""
                ### Variations of network accuracy with respect to changes in _learning rate_, _epochs_ and number of _nodes_
                """)

elif option == ls[3]:

    st.write('The image that the network has in relation to numbers')
    st.write('-------')
    
    col1, col2, col3 = st.columns(3)
   
    col1.image('img/BS_0.png')
    col1.image('img/BS_3.png')
    col1.image('img/BS_6.png')
    col1.image('img/BS_9.png')
    col2.image('img/BS_1.png')
    col2.image('img/BS_4.png')
    col2.image('img/BS_7.png')
    col3.image('img/BS_2.png')
    col3.image('img/BS_5.png')
    col3.image('img/BS_8.png')

elif option == ls[4]:

    st.write("To maintain the integrity of the model's performance on new datasets, it is crucial to optimize it following the established training protocols.")
    st.write("**For this reason, it does not perform well in noisy data**")
    st.write('-------')
    
    a = np.load('saved_weigths/hyper_parameters.npy')
    hn = int(a[0])
    lr = int(a[1])
    inp = 784
    outp = 10
    nn = Neural_Network(inp, hn, outp, lr)
    nn.weigths_input_to_hidden_layer = np.load('saved_weigths/weights_input_to_hidden_layer.npy')
    nn.weigths_hidden_to_output_layer = np.load('saved_weigths/weights_hidden_to_output_layer.npy')
    picture = st.camera_input("Take a picture from digit.", disabled=False)
    from PIL import Image
    if picture:

        picr = Image.open(picture, mode='r').convert('L')
        picc = picr.resize((28,28))
        col1 , col2 = st.columns(2)
        picrr = np.array(picc)
        col = list()

        th = [100,110,120,130,140,150,160,170,180,190]
        col = st.columns(len(th))
        
        for t in th:
            pic = 255 - picrr.reshape(784)
            for i in range(784):
                c = pic[i]
                if c <= t:
                    pic[i] = 0

            outputs = nn.query(pic)
            predicted_number = np.argmax(outputs)
            fig, ax = plt.subplots()
            ax.imshow(pic.reshape((28,28)), cmap='Greys')
            col[th.index(t)].pyplot(fig)
            col[th.index(t)].title(predicted_number)

        st.markdown("""
        #### _Note_: The network does not recognize numbers, it analyzes everything it receives from the input.
                    
        
                    """)