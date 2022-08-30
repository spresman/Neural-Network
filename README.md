# Neural-Network

Network.py implements a neural network from scratch, and is a fantastic way of presenting computational "learning". The network is compatible with any number of layers,
but takes on a three-layer structure (784, 300, 10) to classify hand-written digits from the infamous MNIST dataset. 


![numbers](https://user-images.githubusercontent.com/66577070/187332068-8668d0b2-1645-4cc2-9a4d-622d3dc2fef5.jpg)

# Data
Data is stored as CSV files, and can be found on Kaggle:

https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

Training data: mnist_train.csv                           
Testing data: mnist_test.csv

# Libraries

numpy
Matplotlib

# Familiarization

The instantiated network in Network.py has the structure (784, 300, 10). The first number (784) is the number of nodes in the first layer, or input layer, of the network. The reason for this seemingly random number is that the inputs are 28x28 images of the digits, therefore having 784 pixels (all of which are fed into the network). 

The last number (10), represents the number of nodes in the last layer, or the output layer. The output layer provides us with the answers we are looking for when testing our network. Since we are using the MNIST database, we are interested in the network correctly classifying digits 0-9, which there are 10 of. Each node in the output layer signifies a digit, and the closer the value of a given node with index *n* in the output layer is to 1, the more confident the network is that the input data is the digit *n*. 

The middle number (300), is the number of nodes in the hidden layer. There can be as many hidden layers as we would want, and they serve as somewhat of a "black box" between the input and output nodes. **Note: More hidden layers does not always guarantee a more accurate network**

# Results

After training the network on all the records in mnist_train, we are ready to test our neural network. We can test to see if the network can correctly classify all ten digits from 0 to 9, and will use some data from mnist_test to do so. Let's see how our network does. 

**The closer a node at index *n* is to 1, the more convinced the network is that the inputted digit is *n*!**

![](https://i.imgur.com/Q1hwdEX.png)

This is a zero that our network classified as...

``` 
[0.98124004 0.00795592 0.00686056 0.00749304 0.01078758 0.00712276 0.03091931 0.01781471 0.00576423 0.05786886]
```

a zero! So far, so good. Let's see what the rest of the digits look like. 

----
![](https://i.imgur.com/KsOZaPO.png)

```
[0.00110636 0.99075735 0.00618174 0.00707915 0.01279355 0.00734209 0.01073427 0.01139011 0.00229037 0.0045704 ]
```

----
![](https://i.imgur.com/3K7bqUM.png)

```
[0.04278661 0.01144557 0.91867745 0.07548339 0.00038529 0.00533824 0.01537189 0.00053842 0.00276515 0.00027578]
```
----
![](https://i.imgur.com/L4PLJz3.png)

Wow! That is an ugly three (no offense to whoever scribbled this). I wonder how well our network can handle it!

```
[0.00805419 0.01373759 0.02280802 0.76282533 0.00799402 0.00663105 0.02619592 0.00653008 0.20676145 0.00183157]
```
We can see the network is much less confident in its choice than it was with our previous digits. It seems to still classify it correctly, but it was thinking about it being an 8 too (which I can totally sympathize with).


![](https://i.imgur.com/oUTbQA0.png)

Here's a nicer three. 

```
[0.00647052 0.01042228 0.00162562 0.99047119 0.00325067 0.01010602 0.00041349 0.04147269 0.00029955 0.02019508]
```
The network feels much better about this one. 

----
![](https://i.imgur.com/oPAA63L.png)

```
[0.00224957 0.00757465 0.00361611 0.00067218 0.95091424 0.00503086 0.00431555 0.01231841 0.00347632 0.06768369]
```
----
![](https://i.imgur.com/tt0LbSI.png)

This five is tough even for me, what did the network think?

```
[0.01786709 0.01270842 0.0074122  0.00132361 0.07584937 0.21225323 0.18978295 0.00194908 0.11085246 0.09780113]
```
It wasn't confident at all, but still ended up with the correct classification!


![](https://i.imgur.com/tnSzMJf.png)

Here's a nicer five.
```
[0.00265957 0.02232105 0.00029169 0.15012895 0.00566007 0.95073432 0.00133877 0.00393715 0.01473459 0.00404094]
```
Much better!

----
![](https://i.imgur.com/5yX4RdA.png)

```
[0.01377129 0.00929081 0.01323424 0.01057013 0.00614414 0.0061274 0.87665078 0.0030676  0.13434013 0.0046619 ]
```
----
![](https://i.imgur.com/wRpUPvE.png)

```
[0.01125609 0.00373292 0.00703715 0.00774859 0.00264973 0.01199163 0.00474146 0.98606193 0.0050172  0.00827117]
 ```
----
![](https://i.imgur.com/zOnW42D.png)

```
[0.01817957 0.00295549 0.15859161 0.0035661  0.0087419  0.00938486 0.00791362 0.00316282 0.96733426 0.01915394]
```
----
![](https://i.imgur.com/k0FNcq3.png)

```
[0.00043469 0.00849687 0.01104873 0.01225671 0.0128441  0.0037381 0.00284017 0.0202223  0.00117674 0.98463597]
```

Although the network classified all the digits correctly, a much larger sample size should be tested to obtain the accuracy of the neural network. 
After testing the neural network on all the records in mnist_test (10,000) a total of 10 times, the average performance is 0.9594, or an error of .0406. On average, 
the network classified 96% of the handwritten digits correctly.




