# MLGarden

MLGarden allows you to go from an empty slate to a relatively complex neural network, using a visual editor, no linear algebra, and nothing past high school math. You can play along by downloading a release and building your own networks while you read.

Make sure to play the gifs in this write-up! They don't autoplay on github, the play button is on the top right ⬇️
<p align="center">
<img src="https://github.com/user-attachments/assets/60efbd39-f9ef-4d2e-9052-42242dadb13f" width="700" >
</p>

MLGarden was inspired by watching Karpathy's [spelled-out intro to neural networks](https://www.youtube.com/watch?v=VMj-3S1tku0) and wishing that I could play with the visual computation graphs directly!
<p align="center">
<img src="https://github.com/user-attachments/assets/e22ebfd3-8181-429b-9e46-03fa3b8063e9" width="700" >
</p>

So what can you do in ML Garden? You can make computation graphs such as this one:

<p align="center">
<img src="https://github.com/user-attachments/assets/17505b32-df2d-4c2c-80ad-325c8ece86ab" width="700" >
</p>

All this is, is a different way of expressing (5 + 3) * 2 = 16

Why is this useful? Well, for one, we don't need parenthesis, you just do your calculations from left to right. But there's another advantage: say that we want to change the three parameters, a, b and c, to make the result 10 instead of 16. One way we can do this is by moving each parameter, checking whether the result is going closer or further from 10, and then nudge it in the direction that makes it closer to 10.

<p align="center">
<img src="https://github.com/user-attachments/assets/eaae4a93-26df-4bac-ab3f-54cc1afbe26d" width="700" >
</p>

This is hardly efficient, so how can we systematise this somehow? What we are doing is trying to figure out how the result will change if we change a, b, or c. Another way to put this is that we are figuring out the derivative of the result with respect to each of the parameters. This is where the computation graph comes in handy, through a process called *backpropagation*.

Let's look at a simple case, addition. When we nudge each of the source nodes, the output node moves the same amount. The derivative of the output node with respect to each parameter is 1.

<p align="center">
<img src="https://github.com/user-attachments/assets/148ec9ac-35c3-4dea-8ea9-0cd94dd74b57" width="700" >
</p>

The derivatives are written out in orange.

Lets look at a slightly more complicated case, multiplication. If we nudge the parameters, the amount the result moves depends on the value of the *other* parameter. Easy to see that if b is 0, no matter how much we move a the result won't change. If b is -1 the result will change by the opposite amount we change b. If b is 2, the result will change by twice the amount we change a. We simply copy the value of the other parameter into the derivative of each of the parameters.

<p align="center">
<img src="https://github.com/user-attachments/assets/c3d287df-38f1-4bb7-9d72-ce3d7bc5d4eb" width="700" >
</p>

We can do this for any operation with 2 inputs. You can actually check these on [Wolfram Alpha](https://www.wolframalpha.com/) for some of the more complicated ones, if you don't feel like figuring it out.

<p align="center">
<img src="https://github.com/user-attachments/assets/34ba41c6-9f12-4ca1-8c90-c40ddb27c0ee" width="700" >
</p>

But what if it's a bigger graph, like our initial example? We first start from the result node and calculate the derivatives of its immediate children

<p align="center">
<img src="https://github.com/user-attachments/assets/fcb6db41-2c89-4051-8dc6-ee4c1c283100" width="700" >
</p>

But what do we do about that "a+b" graph. Intuitively, if we change a or b, we know the "a+b" node will change by the same amount, and we know that if the "a+b" node changes, the result node will change by twice that amount. So if we change a, the result will change by twice the amount. Essentially we take the derivative, and we multiply it backwards by the derivative of each individual node. That's back propagation!

<p align="center">
<img src="https://github.com/user-attachments/assets/eff71ad8-69bd-4c06-97ea-fdb1209eb790" width="700" >
</p>

And the benefit of this is that we can keep going, no matter how deep the graph is! We just take a node's derivative value, meaning the result's derivative with respect to that node, and then to get its source nodes derivative we use the two node rules we discussed above, and multiply them by that node's derivative value.

<p align="center">
<img src="https://github.com/user-attachments/assets/a008aafd-ef5f-4abe-a03b-aef81341cda9" width="700" >
</p>

You can scale this up to literally billions of operations! But getting back to our example, how can we make the result 10? We could notice that the result is higher than 10, and therefore nudge all the parameters in the direction that makes the result smaller, until it reaches 10

<p align="center">
<img src="https://github.com/user-attachments/assets/553d174b-a4af-4e4f-9cc7-0aa54f58aa9a" width="700" >
</p>

Can we systematise this for any result we might want? We could create an **error function** that quantifies how wrong we are. A good example here is to take the result of the graph, subtract the desired result (10) from it, and then take the square of that difference, so that value will always go up as the error goes up. Then we could take the derivative of this error, and nudge all the parameters to make the error go down. That's the basis of all deep learning! As long as you can define an error function, you can optimise a large computation graph to lower that error. Let's see what that looks like:

<p align="center">
<img src="https://github.com/user-attachments/assets/3195aa37-b2fd-4227-acf0-42293c1a7b95" width="700" >
</p>


Now that we have our error funtion, we just have to nudge all the parameters by an amount that's the inverse of the derivative. We could for example take the derivative and multiply it by 0.01, and add it to each parameter's value. The 0.01 would be the **learning rate**.

But how is this useful? Basically anything that you can calculate and you can define an error rate for, you can now iteratively improve! If your error rate is how bad you are at predicting the next character in a piece of text, lowering that error rate makes your computation graph better at predicting text, and it turns out that if get good enough at predicting the next token of text Let's start with something simpler, sorting dots into 2 categories:

<p align="center">
<img src="https://github.com/user-attachments/assets/12bb658e-cdea-4d7a-b266-cf7b0cc367e7" width="700" >
</p>

This is our dataset, each dot is a data point, and each point has an x coordinate, a y coordinate, and a label. Our artificial task is to predict, for a given set of coordinates, if the label should be 1- (blue in the image) or 1 (red). So let's define our error function first:

<p align="center">
<img src="https://github.com/user-attachments/assets/e406ba08-2655-41b0-ad87-2c92e966d1f3" width="700" >
</p>

So we will make a prediction, somehow, and then subtract from that prediction the result we know is right (the label), and then square that difference (so that the error is always positive, and so that decreasing the error is always good). Note that there are actually other was of defining the error function, they mostly all work as long as you get a more useful result when the error goes down.

How do we predict this value? Back propagation is pretty forgiving, in this case as long as we have a function that has the ability to describe this dataset, it'll find the right free parameters for it. Now it would be nicer if we didn't have to go in and manually poke the parameters around, so there's a Training window that'll do this process for you, you simply press play and it'll nudge every free paramater by it's derivative to the backprop node (usually the error), go to the next data point, and do it again. As an additional nicety it'll color in the dataset window based on the result node - checking what the result would be for every point in the graph window. Let's try a potential function, just taking the x coordinate of each point, multiplying it by some value, adding some other value to it, and then using [tanh](https://reference.wolfram.com/language/ref/Tanh.html#:~:text=Tanh%20is%20the%20hyperbolic%20tangent,of%20the%20natural%20logarithm%20Log.) to slam it to either -1 or 1.

<p align="center">
<img src="https://github.com/user-attachments/assets/918c9147-a3b7-4804-9be6-5532cf5258d9" width="700" >
</p>

Close, but it seems like the model can only really predict vertical lines - this makes sense, it only knows about each point's x coordinate, it can only represent . Let's instead add in the y coordinate, also multiplied by a parameter, to the x coordinate, and then run the training again.

<p align="center">
<img src="https://github.com/user-attachments/assets/90a0845d-2c17-4229-bccb-1c944c60b0f7" width="700" >
</p>

We did it! Although, we probably could have figured out an equation for this dataset by hand, let's try some tougher ones.

<p align="center">
<img src="https://github.com/user-attachments/assets/0710da57-7f83-416c-aedf-ab1637777017" width="700" >
</p>

A donut. Let's try it with out previous model

<p align="center">
<img src="https://github.com/user-attachments/assets/42dfc791-4fbe-41cd-aaa2-6647a92ddc3a" width="700" >
</p>

Very quickly the model seems to give up, and simply label everything as halfway in between both labels - this makes sense. The model can only express datasets that are split by a line that passes through 0, 0. That's its *inductive bias*. So in this case, what reduces the error is to simply predict 0 for whichever point you pass in. How can we make a model that with the right parameters _can_ represent the data? We could for example feed into the model each point's distance to 0,0, and then do our multiply by a parameter and adding a parameter:

<p align="center">
<img src="https://github.com/user-attachments/assets/929391c3-cca3-402a-9d45-0d9bca20a1c4" width="700" >
</p>

It works! But this feels like cheating - the whole point of this backpropagation business was to not have to figure out things by hand, and now we're designing models based on which shapes we recognise in the data. Is there a class of models that can represent all or most distributions, is scalable, and can be backpropagated through? Hard to tell [who figured it out first](https://en.wikipedia.org/wiki/Backpropagation#History) but inspired by neurons, people figured out that you can make a network of fake, simplified neurons, called perceptrons, where each perceptron takes a bunch of inputs, multiplies each input by some weight, adds a bias, and then applies a non-linearity to the result (like our Tanh). You can then feed these perceptrons into each other, and make a model that can represent any arbitrary function (as long as it has enough perceptrons). Let's see what a single perceptron looks like:

<p align="center">
<img src="https://github.com/user-attachments/assets/27e2072f-0b18-4cce-b674-9ef9686a21f4" width="700" >
</p>

It's actually the exact same network that could predict lines above, the x and the y get multiplied by a parameter each, and then added to another free parameter, and then put through a non-linear function (tanh). Like before, it can't really model this dataset. But what happens if make 2 different perceptrons, and then feed the whole thing into another perceptron?

<p align="center">
<img src="https://github.com/user-attachments/assets/e34025cc-d04c-4daf-8db2-ffbc7cf5fc60" width="700" >
</p>

It already does quite well! but the nice thing about perceptrons is we can keep stacking them, let's add one more:

<p align="center">
<img src="https://github.com/user-attachments/assets/75878ed4-80d3-4a04-96d4-ab4c4df6c869" width="700" >
</p>

It does a little blurry triangle now. This makes some sense, the first "layer" of perceptrons defines a linear boundary each, and then the second perceptron adds them all up to define the shape. Let's add one more and clean it up at little:

<p align="center">
<img src="https://github.com/user-attachments/assets/e7cf8041-a70c-48d9-955e-87112914e0bb" width="700" >
</p>

It gets maybe a little better, but honestly it seems that we have enough perceptrons to do really well at classifying the points in this dataset.

What about this dataset?

<p align="center">
<img src="https://github.com/user-attachments/assets/6020bdc9-d915-4015-acbf-2b3856fa4a35" width="700" >
</p>

As you can see, it doesn't do great. We can try doubling the amount of perceptrons:

<p align="center">
<img src="https://github.com/user-attachments/assets/979d2a37-febf-4c06-9f57-d2a97d9da838" width="700" >
</p>

What if, instead of a single layer with 8, we do 2 layers of 4? This is going to be a lot of connecting:

<p align="center">
<img src="https://github.com/user-attachments/assets/60efbd39-f9ef-4d2e-9052-42242dadb13f" width="700" >
</p>

You can see the second layer perceptrons need 4 inputs each. Results are somewhat better! But we can do better still! Let's do 8 perceptrons in the first layer, and then connect them all to a second layer.

<p align="center">
<img src="https://github.com/user-attachments/assets/a809fd3f-fee4-4b68-85e2-bb924e1ce1e6" width="700" >
</p>

Look on my Works, ye Mighty, and despair! Hopefully by now you're beginning to understand why this isn't how neural networks actually get built. If you count the connections between the first layer and the second, you'll see that there are (4*8) connections, and that this whole web of operations can be neatly described by a 8x4 matrix, but that's exactly what we were not doing. How well does this do though?

<p align="center">
<img src="https://github.com/user-attachments/assets/945f43cb-a4df-4808-8602-d77d6c2c1a1b" width="700" >
</p>

That's satisfying! You can see me fiddle with some of the numbers in the training menu - one of them is the learning rate, how much we judge the parameters' values by each time we look at a sample, the other is the batch size - instead of looking at samples one by one, we look at 50 (and later on 500) and average out all their gradients. For serious neural networks there are training schedules where the learning rate decreases as the model learns, but here we just ping it down whenever, you can see when it needs lowering because the average error (also in the training window) starts ping ponging up and down.

OK, the final boss:

<p align="center">
<img src="https://github.com/user-attachments/assets/56cd9dd8-c766-4ad3-8967-5c751b690056" width="700" >
</p>

Let's see how well the current network does:

<p align="center">
<img src="https://github.com/user-attachments/assets/29574601-1960-47e4-9676-23799d195825" width="700" >
</p>

Not bad! It does take a while to get there so you can start to understand why ML people spend so much time on twitter, but it gets it pretty well. We could of course make the current layers wider, or add more layers, but there's also another old trick up our sleeves, engineering new features! Instead of just passing x and y into the first layer, we could also pass in the distance from some learned point, or the sine of x and y with some offset, for example.

<p align="center">
<img src="https://github.com/user-attachments/assets/2f50dcf2-6677-4ffa-bc5f-2d4e839f391c" width="700" >
</p>

Our new first layer perceptron, with some sines it can use. One last time:

<p align="center">
<img src="https://github.com/user-attachments/assets/5b941afb-647b-4d72-bfa6-e572505142d2" width="700" >
</p>

And how well does it do?

<p align="center">
<img src="https://github.com/user-attachments/assets/6c182db8-cd85-463c-8342-5fabe24d844a" width="700" >
</p>

Pretty good! You might ntice that the model never quite predicts a spiral outside of the spiral that's in the data. As far as the model is concerned, only the points matter, it has no concept of a spiral. The spiral is *your* inductive bias. This is sometimes what people worry about, that if we give a model a ton of examples for what's a good thing to do, when it's presented with a data point that's *out of distribution*, somewhere off the spiral, it won't actually know the pattern.

Go make your own network! You can collapse some of the nodes into functions to make it easier to read, and you can keep on adding new features, more layers, wider layers and seeing what happens. This seems to be what ML people spend a lot of time doing, finding good datasets, with useful error rates, and then figuring out ways to combine that data into new features, add more layers without breaking the training process (as long as you can run backprop on a model it'll learn, but with big networks you can run into issues where the gradients become tiny or huge and everything goes out of control), and hook up giant computers to train bigger models for longer. It mostly seems to be an empirical science, in a paper discussing different activation functions (the tanh we're using is no longer popular, people use functions like [ReLU and SwiGLU](https://medium.com/@jiangmen28/beyond-relu-discovering-the-power-of-swiglu-%E8%B6%85%E8%B6%8A-relu-%E5%8F%91%E7%8E%B0-swiglu-%E7%9A%84%E5%8A%9B%E9%87%8F-9dbc7d8258bf)) the authors describe this incredible ability to learn no matter what as "divine benevolence".

<p align="center">
<img src="https://github.com/user-attachments/assets/1e3e3428-5105-41b1-94e2-1c5fd8a14cda" width="700" >
</p>

I do think there's something divine about it - measuring and nudging and eventually creating something incredibly powerful - god knows where it'll take us.

## How to build

```
git submodule update --init --recursive
mkdir build
cd build
cmake ..
```
